import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

from imbox import Imbox


def download_inbox(
    output_path: Path,
    host: str,
    port: int,
    username: str,
    password: str,
    ssl: bool,
    folder: str,
    label: str,
) -> Path:
    if output_path.exists():
        logging.info("Messages already downloaded, not downloading again.")
        logging.info(f"(Delete `{output_path}` to re-download.)")
        return output_path

    with Imbox(
        hostname=host,
        port=port,
        username=username,
        password=password,
        ssl=ssl,
    ) as imbox:

        def message_to_json(uid, message) -> Dict[str, Any]:
            return {
                "uid": uid.decode(),
                "subject": message.subject,
                "body": "".join(message.body["plain"]),
                "from": message.sent_from,
            }

        def process_messages(iter) -> Iterable[Dict[str, Any]]:
            for (uid, message) in iter:
                try:
                    logging.info("Processing {}: {}".format(uid, message.subject))
                    yield message_to_json(uid, message)
                except Exception as e:
                    logging.exception(f"Failed to process message: {message!r}", e)

        spam_messages = list(
            process_messages(imbox.messages(folder=folder, label=label))
        )
        all_messages = list(process_messages(imbox.messages(folder=folder)))

    data = json.dumps(
        {
            "all_messages": all_messages,
            "spam_messages": spam_messages,
        },
    )
    with open(output_path, "w") as f:
        f.write(data)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, type=Path, help="The output path to write to"
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Hostname of the IMAP server (such as imap.gmail.com)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the IMAP server (such as 143 or 993)",
    )
    parser.add_argument("--username", required=True, help="The IMAP username")
    parser.add_argument("--password", required=True, help="The IMAP password")
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Whether or not to use SSL",
    )
    parser.add_argument(
        "--folder", default="all", help="The folder to download messages from"
    )
    parser.add_argument(
        "--label",
        default="recruiter-spam",
        help="The label marking emails as recruiter spam",
    )
    args: argparse.Namespace = parser.parse_args()
    download_inbox(
        output_path=args.output,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        ssl=args.ssl,
        folder=args.folder,
        label=args.label,
    )


if __name__ == "__main__":
    main()
