"""Shared exact scientific helpers for the native controls."""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from dataclasses import fields


def _pack(record, arguments):
    for field in fields(arguments):
        setattr(record, field.name, getattr(arguments, field.name))
    return record
