"""Module entrypoint: viam-server starts this through run.sh."""

import asyncio

from viam.module.module import Module

from .service import BowlContents

if __name__ == "__main__":
    asyncio.run(Module.run_with_models(BowlContents))
