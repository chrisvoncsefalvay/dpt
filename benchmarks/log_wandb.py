"""Upload saved measurements and figures after timing to a private W&B project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="Your W&B account")
    parser.add_argument("--label", required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [(path, json.loads(path.read_text())) for path in args.results]

    import wandb
    from wandb.apis._generated import CREATE_PROJECT_GQL, UpsertModelInput

    api = wandb.Api()
    request = UpsertModelInput(name="dpt-performance", entity_name=args.entity, access="PRIVATE")
    reply = api._service_api.execute_graphql(CREATE_PROJECT_GQL, {"input": request.model_dump()})
    project = reply["result"]["project"]
    if project["access"] != "PRIVATE":
        raise RuntimeError("Project privacy could not be verified; nothing was uploaded")
    config = {
        "measurement_files": [str(path) for path in args.results],
        "measurement_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.results
        },
        "privacy": "PRIVATE",
        "timings_collected_before_wandb_run": True,
    }
    with wandb.init(
        entity=args.entity,
        project="dpt-performance",
        name=args.label,
        config=config,
        job_type="measured-benchmark",
    ) as run:
        table = wandb.Table(
            columns=[
                "file",
                "operation",
                "device_median_ms",
                "device_p95_ms",
                "wall_median_ms",
                "wall_p95_ms",
            ]
        )
        artifact = wandb.Artifact(args.label + "-measurements", type="benchmark")
        for path, data in documents:
            artifact.add_file(str(path), name=path.parent.name + "/" + path.name)
            for row in data.get("measurements", []):
                table.add_data(
                    path.parent.name,
                    row.get("label", row.get("name")),
                    *(
                        row.get(key)
                        for key in (
                            "device_median_ms",
                            "device_p95_ms",
                            "wall_median_ms",
                            "wall_p95_ms",
                        )
                    ),
                )
        run.log({"measurements": table})
        run.log_artifact(artifact)
        for directory, kind in ((args.profiles, "nsight"), (args.figures, "figures")):
            if directory:
                artifact = wandb.Artifact(args.label + "-" + kind, type=kind)
                artifact.add_dir(str(directory))
                run.log_artifact(artifact)
        if args.figures:
            run.log({path.stem: wandb.Image(str(path)) for path in args.figures.glob("*.png")})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "run_url": run.url,
                    "project_access": project["access"],
                    "run_id": run.id,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
