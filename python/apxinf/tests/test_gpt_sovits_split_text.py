"""Official TextPreprocessor.pre_seg_text post-processing through _split_text.

The segmentation stub below is copied verbatim from the official
RVC-Boss/GPT-SoVITS `GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py`
(https://github.com/RVC-Boss/GPT-SoVITS), so the
policy's post-processing runs against the real cut methods, split-punctuation
set and split_big_text. The merge helper in the policy module is copied
verbatim from the official TextPreprocessor.py.
"""
import unittest

import numpy as np

from apxinf import GptSovitsPolicy
from apxinf.policies.impls.gpt_sovits import merge_short_text_in_array


class OfficialSegmentation:
    """Verbatim copy of the official text_segmentation_method.py."""

    punctuation = set(["!", "?", "…", ",", ".", "-", " "])
    splits = {
        "，",
        "。",
        "？",
        "！",
        ",",
        ".",
        "?",
        "!",
        "~",
        ":",
        "：",
        "—",
        "…",
    }

    def split_big_text(self, text, max_len=510):
        import re

        punctuation = "".join(self.splits)
        segments = re.split("([" + punctuation + "])", text)
        result = []
        current_segment = ""
        for segment in segments:
            if len(current_segment + segment) > max_len:
                result.append(current_segment)
                current_segment = segment
            else:
                current_segment += segment
        if current_segment:
            result.append(current_segment)
        return result

    def split(self, todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def _register(self):
        return {
            "cut0": self.cut0,
            "cut1": self.cut1,
            "cut2": self.cut2,
            "cut3": self.cut3,
            "cut4": self.cut4,
            "cut5": self.cut5,
        }

    def cut0(self, inp):
        if not set(inp).issubset(self.punctuation):
            return inp
        else:
            return "\n"

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut2(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        if len(opts) > 1 and len(opts[-1]) < 50:
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut3(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut4(self, inp):
        import re

        inp = inp.strip("\n")
        opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
        mergeitems = []
        items = []
        for i, char in enumerate(inp):
            if char in punds:
                if (
                    char == "."
                    and i > 0
                    and i < len(inp) - 1
                    and inp[i - 1].isdigit()
                    and inp[i + 1].isdigit()
                ):
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)
        if items:
            mergeitems.append("".join(items))
        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)

    def get_method_names(self):
        return list(self._register())

    def get_method(self, name):
        methods = self._register()
        if name not in methods:
            raise ValueError(f"Method {name} not found")
        return methods[name]


class Runner:
    sample_rate = 48000
    device = "cpu"
    is_prepared = False


class SplitTextTests(unittest.TestCase):
    def setUp(self):
        self.policy = GptSovitsPolicy(
            Runner(), frontend={"segmentation": OfficialSegmentation()}
        )

    def test_cut0_long_text_is_split_within_the_bert_window(self):
        # 600 characters of one punctuation-free run used to reach BERT as a
        # single fragment and exceed its 512-position window.
        text = "你好世界。" * 120
        fragments = self.policy._split_text(text, "cut0")
        self.assertGreaterEqual(len(fragments), 2)
        for fragment in fragments:
            self.assertLessEqual(len(fragment), 510)
        self.assertEqual("".join(fragments), text)

    def test_cut1_merges_short_fragments(self):
        # The first clause "一" is shorter than four characters and does not
        # start with split punctuation, so pre_seg_text prefixes "。" before
        # cut1 runs; the trailing 2-character sentence is below the merge
        # threshold and folds into the previous group.
        self.assertEqual(
            self.policy._split_text("一。二。三。四。五", "cut1"),
            ["。一。二。三。四。五。"],
        )

    def test_cut5_merges_short_fragments_and_appends_punctuation(self):
        # pre_seg_text prefixes "。" ("火" opens a sub-threshold first clause),
        # but cut5's own punctuation filter drops the lone "。" item, so the
        # prefix does not reach the output: cut5 flushes one fragment per
        # punctuation ("火。", "水。", "土"), the official merge folds each
        # short neighbour in turn, and the result receives a terminal split
        # punctuation because "土" does not end in one.
        self.assertEqual(self.policy._split_text("火。水。土", "cut5"), ["火。水。土。"])

    def test_cut5_appends_terminal_punctuation(self):
        self.assertEqual(self.policy._split_text("你好世界", "cut5"), ["你好世界。"])

    def test_cut3_appends_terminal_punctuation(self):
        # cut3 strips the sentence-final 。 from each line, so the merged
        # fragment ends without one and the post-processing restores it.
        self.assertEqual(self.policy._split_text("甲。乙", "cut3"), ["甲乙。"])

    def test_cut4_appends_terminal_punctuation(self):
        # pre_seg_text prefixes "。" (the zh first clause "ab" is
        # sub-threshold); cut4's strip(".") does not remove it and the prefix
        # merges into the fragment before the terminal punctuation.
        self.assertEqual(self.policy._split_text("ab.cd", "cut4"), ["。abcd。"])

    def test_cut2_splits_at_fifty_characters(self):
        # pre_seg_text prefixes "。" (the first clause "一" is
        # sub-threshold); cut2 then groups past fifty characters from
        # there: "。" plus the first 25 "一。" flush first, and the
        # remainder carries 5 "一。" and all 30 "二。".
        text = "一。" * 30 + "二。" * 30
        self.assertEqual(
            self.policy._split_text(text, "cut2"),
            ["。" + "一。" * 25, "一。" * 5 + "二。" * 30],
        )

    def test_short_first_clause_gets_a_leading_punctuation(self):
        # pre_seg_text: a first clause shorter than four characters that does
        # not start with split punctuation receives one ("。" for Chinese,
        # "." for English) before the segmentation method runs.
        self.assertEqual(self.policy._split_text("你好", "cut0"), ["。你好。"])
        self.policy.language = "en"
        self.assertEqual(self.policy._split_text("ab", "cut0"), [".ab."])

    def test_punctuation_only_lines_are_skipped(self):
        # "、" is not in the cut methods' own half-width punctuation filter
        # ({"!","?","…",",",".","-"," "}), so the punctuation-only line
        # survives cut3 and the merge; pre_seg_text's re.sub(r"\W+", ...) is
        # what drops it before terminal punctuation is appended.
        fragments = self.policy._split_text("甲乙丙丁戊。、、、、、。", "cut3")
        self.assertEqual(fragments, ["甲乙丙丁戊。"])
        self.assertNotIn("、、、、、", fragments)

    def test_all_blank_lines_raise_value_error(self):
        # cut3 filters "。。" down to an empty string; filter_text receives
        # [""] and raises, as the official pre_seg_text does.
        with self.assertRaises(ValueError) as ctx:
            self.policy._split_text("。。", "cut3")
        self.assertIn("请输入有效文本", str(ctx.exception))

    def test_english_text_receives_an_english_terminal_punctuation(self):
        self.policy.language = "en"
        self.assertEqual(
            self.policy._split_text("hello world", "cut5"), ["hello world."]
        )

    def test_merge_short_text_in_array_edges(self):
        # Fewer than two fragments pass through untouched; a final
        # sub-threshold remainder joins the last flushed fragment.
        self.assertEqual(merge_short_text_in_array(["甲。"], 5), ["甲。"])
        self.assertEqual(
            merge_short_text_in_array(["甲乙。", "丙"], 5), ["甲乙。丙"]
        )
        self.assertEqual(merge_short_text_in_array(["甲", "乙"], 5), ["甲乙"])


class RecordingRunner(Runner):
    def __init__(self):
        self.calls = []

    def synthesize(self, phone_ids, *args, **kwargs):
        self.calls.append(phone_ids)
        return {
            "samples": np.zeros(16, dtype=np.float32),
            "valid_length": 16,
            "sample_rate": 48000,
            "complete": True,
        }


class PublicInferTests(unittest.TestCase):
    def make(self):
        runner = RecordingRunner()
        policy = GptSovitsPolicy(
            runner, frontend={"segmentation": OfficialSegmentation()}
        )
        policy._text_inputs = lambda text, reference=False: (text, [0], [1])
        policy._reference_inputs = lambda audio: (
            np.zeros(4, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        return policy, runner

    def observation(self, text, method="cut0"):
        return {
            "text": text,
            "reference_text": "参考文本。",
            "reference_audio": object(),
            "text_split_method": method,
        }

    def test_cut0_splits_on_newlines_like_the_official_postpass(self):
        paragraph = "你好世界" * 75
        policy, runner = self.make()
        result = policy.infer(self.observation(paragraph + "\n" + paragraph))
        self.assertEqual(
            [fragment["text"] for fragment in result["fragments"]],
            [paragraph + "。", paragraph + "。"],
        )
        self.assertEqual(len(runner.calls), 2)

    def test_cut0_short_newline_text_is_one_fragment_per_line(self):
        policy, _ = self.make()
        result = policy.infer(self.observation("今天天气很好。\n明天会下雨吧。"))
        self.assertEqual(len(result["fragments"]), 2)

    def test_empty_text_is_rejected_before_synthesis(self):
        for method in ("cut0", "cut5"):
            for text in ("", "\n", "\n\n\n"):
                policy, runner = self.make()
                with self.assertRaises(ValueError) as ctx:
                    policy.infer(self.observation(text, method))
                self.assertIn("no valid fragment", str(ctx.exception))
                self.assertEqual(runner.calls, [])

    def test_prepare_rejects_empty_text(self):
        policy, _ = self.make()
        with self.assertRaises(ValueError):
            policy.prepare(self.observation(""), target_frames=4)

    def test_split_text_keeps_the_official_empty_result(self):
        policy, _ = self.make()
        self.assertEqual(policy._split_text("", "cut0"), [])
        self.assertEqual(policy._split_text("\n\n", "cut5"), [])


if __name__ == "__main__":
    unittest.main()
