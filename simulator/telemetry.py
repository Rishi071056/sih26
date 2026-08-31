import csv
import json
import os
from datetime import datetime


class TelemetryLogger:

    def __init__(self, csv_path: str):

        self.csv_path = csv_path

        directory = os.path.dirname(
            csv_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        self.file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = None

    def write(self, telemetry: dict):

        if self.writer is None:

            self.writer = csv.DictWriter(
                self.file,
                fieldnames=telemetry.keys()
            )

            self.writer.writeheader()

        self.writer.writerow(
            telemetry
        )

        self.file.flush()

    def close(self):

        self.file.close()


def telemetry_to_json(
    telemetry: dict
) -> str:

    return json.dumps(
        telemetry,
        separators=(",", ":")
    )