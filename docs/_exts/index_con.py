"""Convert the repository README into the legacy Sphinx index page."""

import os

from m2r2 import convert


class IndexCon:
    def __init__(self, source: str, output: str = "index.rst") -> None:
        self.source = source
        self.output = output
        self.preprocess()

    def preprocess(self) -> None:
        with open(self.source, encoding="utf-8") as f:
            # remove the CI link from the file
            lines = f.readlines()
            lines = [line for line in lines if "[CI]" not in line]

            # change local links to the ones related to the _build/html directory and extension to .html
            lines = [line.replace("](docs/", "](") for line in lines]
            lines = [line.replace(".md)", ".html)") for line in lines]

            readme_text = "".join(lines)
            result = convert(readme_text)
            if (
                "<details>" in readme_text
                and "<summary>" in readme_text
                and "</details>" in readme_text
            ):
                start_details_tag = [line for line in lines if "<details>" in line]
                summary_tag = [line for line in lines if "<summary>" in line]
                end_details_tag = [line for line in lines if "</details>" in line]
                start_details = lines.index(start_details_tag[0])
                summary_line = lines.index(summary_tag[0])
                end_details = lines.index(end_details_tag[0])

                before = convert("".join(lines[: start_details - 1]))
                end = convert("".join(lines[end_details + 1 :]))

                collapse_rst = lines[summary_line + 1 : end_details]
                collapse_rst = [
                    "**" + x.split("# ")[1][:-1] + "**\n" if "# " in x else x for x in collapse_rst
                ]
                collapse_rst = convert("".join(collapse_rst))
                collapse_rst = collapse_rst.split("\n")
                collapse_rst = ["    " + x for x in collapse_rst]
                collapse_rst = ["\n.. collapse:: Click to SHOW examples\n"] + collapse_rst
                result = before + "\n".join(collapse_rst) + end

            if os.path.exists(self.output):
                os.remove(self.output)

            with open(self.output, "a", encoding="utf-8") as f:
                f.write(result)
                f.write("\n\n")

                toc_dir = os.path.dirname(self.output)
                toc_path = os.path.join(toc_dir, "toc.rst")
                if not os.path.exists(toc_path):
                    toc_path = os.path.join(toc_dir, "toc.bak")
                if os.path.exists(toc_path):
                    with open(toc_path, encoding="utf-8") as t:
                        f.write(t.read())


if __name__ == "__main__":
    index = IndexCon("../../README.md")
