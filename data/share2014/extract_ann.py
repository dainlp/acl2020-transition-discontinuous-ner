import argparse, logging, os, re, sys

logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--ann_dir", default=None, type=str)
    parser.add_argument("--text_dir", default=None, type=str)
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--log_filepath", default="output.log", type=str)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_parameters()
    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    mentions = {}
    for filename in os.listdir(args.ann_dir):
        with open(os.path.join(args.text_dir, filename.replace(".pipe", ""))) as text_f:
            text = text_f.read()
            with open(os.path.join(args.ann_dir, filename)) as in_f:
                for line in in_f:
                    sp = line.strip().split("|")
                    assert sp[0] == filename.replace(".pipe", "")
                    indices = re.findall(r"\d+", sp[1])
                    indices = sorted([int(i) for i in indices])
                    mention_text, gap_text = [], []
                    for i in range(0, len(indices), 2):
                        mention_text.append(
                            text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())
                    for i in range(1, len(indices) - 2, 2):
                        gap_text.append(text[indices[i]:indices[i + 1]].replace("\n", " ").replace("\t", " ").strip())

                    mention_text = " ".join(mention_text).strip()
                    gap_text = " ".join(gap_text).strip()
                    if len(indices) > 2 and len(gap_text) == 0:
                        logger.info("%s in %s is not a real discontinuous entity, as the mention (%s) has no gap." % (
                        ",".join([str(i) for i in indices]), filename, mention_text))
                        indices = [indices[0], indices[-1]]
                    str_indices = ",".join([str(i) for i in indices])
                    mentions[(filename.replace(".pipe", ""), str_indices)] = (mention_text, gap_text)

    with open("%s.ann" % args.split, "w") as out_f:
        for k, v in mentions.items():
            out_f.write("%s\tDisorder\t%s\t%s\t%s\n" % (k[0], k[1], v[0], v[1]))