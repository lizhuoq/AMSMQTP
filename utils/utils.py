import zipfile
import os
from os.path import join
from os import listdir
import asyncio
import time

import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
# import cdsapi


def unzip(zip_file_path, extract_to_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(
            join(extract_to_folder, zip_file_path.split('/')[-1].split('.')[0])
        )


async def request_wait(sleep, r):
    while True:
        r.update()
        reply = r.reply
        r.info("Request ID: %s, state: %s" % (reply["request_id"], reply["state"]))

        if reply["state"] == "completed":
            break
        elif reply["state"] in ("queued", "running"):
            r.info("Request ID: %s, sleep: %s", reply["request_id"], sleep)
            await asyncio.sleep(sleep)
        elif reply["state"] in ("failed",):
            r.error("Message: %s", reply["error"].get("message"))
            r.error("Reason:  %s", reply["error"].get("reason"))
            for n in (
                reply.get("error", {}).get("context", {}).get("traceback", "").split("\n")
            ):
                if n.strip() == "":
                    break
                r.error("  %s", n)
            raise Exception(
                "%s. %s." % (reply["error"].get("message"), reply["error"].get("reason"))
            )
