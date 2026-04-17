import logging
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config.settings import SCHEDULER_TIMEZONE
from db.database import init_db
from ingestion.rss_scraper import fetch_all_rss
from ingestion.reddit_scraper import fetch_all_reddit
from ingestion.ndtv_scraper import fetch_ndtv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def scrape_all() -> None:
    """Run all scrapers in sequence. Called by every scheduler job."""
    started = datetime.now(timezone.utc).isoformat()
    log.info(f"scrape_all() started at {started}")

    rss_new     = fetch_all_rss()
    reddit_new  = fetch_all_reddit()
    ndtv_new    = fetch_ndtv()

    try:
        from ingestion.marketaux_scraper import fetch_marketaux
        marketaux_new = fetch_marketaux()
    except Exception as e:
        log.error(f"MarketAux scrape failed: {e}")
        marketaux_new = 0

    total = rss_new + reddit_new + ndtv_new + marketaux_new
    log.info(
        f"scrape_all() done — RSS: {rss_new}, Reddit: {reddit_new}, "
        f"NDTV: {ndtv_new}, MarketAux: {marketaux_new}, Total new: {total}"
    )


def scrape_and_process() -> None:
    """Run ingestion followed by NLP pipeline."""
    scrape_all()
    from pipeline.orchestrator import run_full_pipeline
    result = run_full_pipeline()
    log.info(f"Scrape + process complete: {result}")


def _job_listener(event):
    if event.exception:
        log.error(f"Job {event.job_id} raised an exception: {event.exception}")
    else:
        log.info(f"Job {event.job_id} completed successfully.")


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_listener(_job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    # Every 15 minutes during NSE market hours (Mon–Fri, 09:00–16:00 IST)
    scheduler.add_job(
        scrape_and_process, "cron",
        id="market_hours",
        day_of_week="mon-fri",
        hour="9-16",
        minute="*/15",
        timezone=SCHEDULER_TIMEZONE,
    )

    # Every hour outside market hours on weekdays (Mon–Fri, 00:00–08:59 and 16:00–23:59 IST)
    scheduler.add_job(
        scrape_and_process, "cron",
        id="off_hours_weekday",
        day_of_week="mon-fri",
        hour="0-8,16-23",
        minute="0",
        timezone=SCHEDULER_TIMEZONE,
    )

    # Every hour on weekends
    scheduler.add_job(
        scrape_and_process, "cron",
        id="weekend",
        day_of_week="sat,sun",
        minute="0",
        timezone=SCHEDULER_TIMEZONE,
    )

    return scheduler


if __name__ == "__main__":
    init_db()
    log.info("Running one immediate scrape_and_process() before starting the scheduler...")
    scrape_and_process()

    log.info("Starting scheduler. Press Ctrl+C to stop.")
    scheduler = build_scheduler()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")
