import argparse, logging, sys

logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--input_filepath", default=None, type=str)
    parser.add_argument("--log_filepath", default="output.log", type=str)

    args, _ = parser.parse_known_args()
    return args


def _load_token_boundaries(filepath):
    token_start, token_end = {}, {}
    with open(filepath) as f:
        pre_doc, sentence_idx, token_idx = None, 0, 0
        for line in f:
            if len(line.strip()) == 0:
                sentence_idx += 1
                token_idx = 0
                continue
            sp = line.strip().split()
            assert len(sp) == 4
            cur_doc = sp[1]
            if pre_doc is None:
                pre_doc = cur_doc
            if pre_doc != cur_doc:
                sentence_idx = 0
                assert token_idx == 0
                pre_doc = cur_doc
            start, end = int(sp[2]), int(sp[3])
            token_start[(cur_doc, start)] = (sentence_idx, token_idx, sp[0])
            token_end[(cur_doc, end)] = (sentence_idx, token_idx, sp[0])
            token_idx += 1
    return token_start, token_end


def _find_token_idx(document, char_offset, token_boundaries, start=True):
    if (document, char_offset) in token_boundaries: return (token_boundaries[(document, char_offset)], 0)

    for offset_adjust in range(1, 10):
        if start:
            if (document, char_offset + offset_adjust) in token_boundaries:
                return (token_boundaries[(document, char_offset + offset_adjust)], offset_adjust)
            if (document, char_offset - offset_adjust) in token_boundaries:
                return (token_boundaries[(document, char_offset - offset_adjust)], -offset_adjust)
        else:
            if (document, char_offset - offset_adjust) in token_boundaries:
                return (token_boundaries[(document, char_offset - offset_adjust)], -offset_adjust)
            if (document, char_offset + offset_adjust) in token_boundaries:
                return (token_boundaries[(document, char_offset + offset_adjust)], offset_adjust)

    logger.info("Cannot find token whose %s offset is %s from document %s." % ("start" if start else "end", char_offset, document))
    return (None, None)


if __name__ == "__main__":
    args = parse_parameters()
    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    for split in ["train", "test"]:
        token_start, token_end = _load_token_boundaries("%s.tokens" % split)
        num_orig_ann, num_final_ann = 0, 0
        with open("%s.token.ann" % split, "w") as out_f:
            with open("%s.ann" % split) as in_f:
                for line in in_f:
                    sp = line.strip().split("\t")
                    assert len(sp) == 5 or len(sp) == 4, sp
                    document, label, indices, mention = sp[0:4]
                    indices = [int(i) for i in indices.split(",")]
                    num_orig_ann += 1

                    token_idx = []
                    for i in range(0, len(indices), 2):
                        start, end = indices[i], indices[i + 1]
                        start_token_idx, offset_adjust = _find_token_idx(document, start, token_start, True)
                        if start_token_idx is not None:
                            token_idx.append(start_token_idx)
                            if offset_adjust != 0:
                                logger.info("Find token whose original start offset is %s by adjusting its offset %s." % (start, offset_adjust))
                        end_token_idx, offset_adjust = _find_token_idx(document, end, token_end, False)
                        if end_token_idx is not None:
                            token_idx.append(end_token_idx)
                            if offset_adjust != 0:
                                logger.info("Find token whose original end offset is %s by adjusting its offset %s." % (end, offset_adjust))

                    if len(token_idx) != len(indices):
                        logger.info("Cannot find all corresponding token indices, so abandon this annotation from document %s" % document)
                    else:
                        if len(set([i[0] for i in token_idx])) != 1:
                            logger.info("This annotation from document %s is abandoned because it crossing multiple sentences." % document)
                        else:
                            sentence_idx = token_idx[0][0]
                            token_idx = [str(i[1]) for i in token_idx]
                            num_final_ann += 1
                            out_f.write("%s\t%s\t%s\t%s\t%s\n" % (document, sentence_idx, label, ",".join(token_idx), mention))

        logger.info("Convert %d out of %d annotations." % (num_final_ann, num_orig_ann))