import argparse, logging, os, sys
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default=None, type=str)
    parser.add_argument("--text_dir", default=None, type=str)
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--log_filepath", default="output.log", type=str)

    args, _ = parser.parse_known_args()
    return args


def process_test(input_dir, text_dir, output_filepath):
    mentions = {}

    for filename in os.listdir(input_dir):
        with open(os.path.join(text_dir, filename)) as text_f:
            text = text_f.read()
            with open(os.path.join(input_dir, filename)) as in_f:
                for line in in_f:
                    sp = line.strip().split("||")
                    assert sp[1] == "Disease_Disorder" and sp[0] == filename
                    indices = [i for i in sp[3:]]
                    indices = sorted([int(i) for i in indices])
                    mention_text, gap_text = [], []
                    for i in range(0, len(indices), 2):
                        mention_text.append(text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())
                    for i in range(1, len(indices) - 2, 2):
                        gap_text.append(text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())
                    mention_text = " ".join(mention_text).strip()
                    gap_text = " ".join(gap_text).strip()
                    if len(indices) > 2 and len(gap_text) == 0:
                        logger.info("%s is not a real discontinuous entity, as the mention (%s) has no gap." % (
                        line.strip(), mention_text))
                        indices = [indices[0], indices[-1]]
                    str_indices = ",".join([str(i) for i in indices])
                    mentions[(filename, str_indices)] = (mention_text, gap_text)

    with open(output_filepath, "w") as f:
        for k, v in mentions.items():
            f.write("%s\tDisorder\t%s\t%s\t%s\n" % (k[0], k[1], v[0], v[1]))


def process_train(input_dir, text_dir, output_filepath):
    mentions = {}

    for filename in os.listdir(input_dir):
        with open(os.path.join(text_dir, filename.replace(".knowtator.xml", "")), "r") as text_f:
            text = text_f.read()
            root = ET.parse(os.path.join(input_dir, filename)).getroot()
            document = root.get("textSource")

            assert document.find("DISCHARGE_SUMMARY") > 0 or document.find("ECHO_REPORT") > 0 or document.find(
                "RADIOLOGY_REPORT") > 0 or document.find("ECG_REPORT") > 0

            for mention in root.findall("annotation"):
                indices = []
                for span in mention.findall("span"):
                    indices.append(span.get("start"))
                    indices.append(span.get("end"))
                indices = sorted([int(i) for i in indices])
                mention_text, gap_text = [], []
                for i in range(0, len(indices), 2):
                    mention_text.append(text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())
                for i in range(1, len(indices) - 2, 2):
                    gap_text.append(text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())

                mention_text = " ".join(mention_text).strip()
                gap_text = " ".join(gap_text).strip()
                if len(indices) > 2 and len(gap_text) == 0:
                    logger.info("%s in %s is not a real discontinuous entity, as the mention (%s) has no gap." % (
                    ",".join([str(i) for i in indices]), filename, mention_text))
                    indices = [indices[0], indices[-1]]
                str_indices = ",".join([str(i) for i in indices])
                mentions[(document, str_indices)] = (mention_text, gap_text)

    with open(output_filepath, "w") as f:
        for k, v in mentions.items():
            f.write("%s\tDisorder\t%s\t%s\t%s\n" % (k[0], k[1], v[0], v[1]))


if __name__ == "__main__":
    args = parse_parameters()
    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    assert args.split in ["train", "test"]
    if args.split == "train":
        process_train(args.input_dir, args.text_dir, "%s.ann" % args.split)
    else:
        process_test(args.input_dir, args.text_dir, "%s.ann" % args.split)