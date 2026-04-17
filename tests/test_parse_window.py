import pytest
from fastapi import HTTPException
from api.main import parse_window

def test_preset_24h():
    hours, decay = parse_window("24h")
    assert hours == 24 and decay == 0.30

def test_preset_7d():
    hours, decay = parse_window("7d")
    assert hours == 168 and decay == 0.05

def test_custom_48h():
    hours, decay = parse_window("48h")
    assert hours == 48 and decay > 0

def test_custom_1h():
    hours, decay = parse_window("1h")
    assert hours == 1

def test_invalid_format():
    with pytest.raises(HTTPException):
        parse_window("invalid")
        

def test_out_of_range():
    with pytest.raises(HTTPException):
        parse_window("9000h")