import argparse
import json
import logging
import mailbox
import quopri
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _get_labels(message: mailbox.Message) -> Set[str]:
    if "X-Gmail-Labels" in message:
        return set(message["X-Gmail-Labels"].split(","))
    return set()


def _get_body(message: mailbox.Message) -> Optional[str]:
    # https://stackoverflow.com/a/1463144
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload()
                break
        else:
            return None
    else:
        payload = message.get_payload()

    # I don't know why this decoding isn't done by the mailbox class.
    return quopri.decodestring(payload).decode()


def unpack_mbox(mbox_path: Path, output_path: Path, folder: str, label: str) -> None:
    if output_path.exists():
        logging.info("Messages already unpacked, not unpacked again.")
        logging.info(f"(Delete `{output_path}` to repopulate.)")
        return

    logging.info(
        "Note: the --folder option is currently ignored (current value: %s)", folder
    )
    logging.info("Loading .mbox at path: %s", mbox_path)
    box = mailbox.mbox(mbox_path, create=False)

    all_messages: List[Dict[str, Any]] = []
    spam_messages: List[Dict[str, Any]] = []
    message: mailbox.Message
    pass_count = 0
    fail_count = 0
    for key, message in box.items():
        try:
            labels = _get_labels(message)
            subject = str(message["Subject"])
            from_ = str(message["From"])
            body = _get_body(message)
            if body is None:
                logging.warn("Could not extract body for message %s", subject)
                continue

            payload = {
                "uid": key,
                "subject": subject,
                "body": body,
                "from": from_,
            }
            all_messages.append(payload)
            if label in labels:
                spam_messages.append(payload)
            logging.info("Processed message: %s", subject)
            pass_count += 1
        except Exception as e:
            fail_count += 1
            logging.exception("Couldn't parse message: %s", repr(message), e)

    logging.info(
        "Parsed %d messages, failed to parse %d messages", pass_count, fail_count
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "all_messages": all_messages,
                "spam_messages": spam_messages,
            },
            f,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mbox", type=Path, required=True, help="The .mbox file to use"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="The output path to write to"
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
    unpack_mbox(
        mbox_path=args.mbox,
        output_path=args.output,
        folder=args.folder,
        label=args.label,
    )


if __name__ == "__main__":
    main()
