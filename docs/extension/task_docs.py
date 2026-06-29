"""
Sphinx extension to auto-generate task descriptions from task_table.tsv. This way we dont have to manually add tasks.
The only thing that can be added manually (optional) is media for each task. Just go to images/ and add
{task_name}.png, {task_name}_2.png, ... for screenshots, and/or {task_name}.mp4, {task_name}_2.mp4, ... for short
demo videos. PNGs render as images; MP4s render as an inline HTML5 video player.

Usage in an .rst file:

    .. task-descriptions::

    .. task-summary-table::
"""

import csv
import json
import os
import shutil
from pathlib import Path
from docutils import nodes
from docutils.statemachine import StringList
from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective


def _read_task_table(tsv_path):
    """Read the task_table.tsv and return a list of dicts."""
    tasks = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("name", "").strip():
                continue
            tasks.append(row)
    return tasks


def _read_task_details(json_path):
    """Read the task_details.json and return a dict keyed by task name."""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_task_media(task_name, media_dir, ext):
    """Find media for a task with the given extension: {name}{ext}, {name}_2{ext}, ..."""
    found = []
    primary = media_dir / f"{task_name}{ext}"
    if primary.exists():
        found.append(primary)

    n = 2
    while True:
        variant = media_dir / f"{task_name}_{n}{ext}"
        if variant.exists():
            found.append(variant)
            n += 1
        else:
            break

    return found


def _find_task_images(task_name, images_dir):
    """Find images for a task: {name}.png, {name}_2.png, {name}_3.png, ..."""
    return _find_task_media(task_name, images_dir, ".png")


def _find_task_videos(task_name, images_dir):
    """Find demo videos for a task: {name}.mp4, {name}_2.mp4, {name}_3.mp4, ..."""
    return _find_task_media(task_name, images_dir, ".mp4")


class TaskDescriptionsDirective(SphinxDirective):
    """Directive that generates task description sections from task_table.tsv."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        source_dir = Path(self.env.srcdir)
        tsv_path = source_dir.parent / "MultiTaskBattery" / "task_table.tsv"
        json_path = source_dir.parent / "MultiTaskBattery" / "task_details.json"
        images_dir = source_dir / "images"

        self.env.note_dependency(str(tsv_path))
        self.env.note_dependency(str(json_path))

        tasks = _read_task_table(tsv_path)
        details = _read_task_details(json_path)

        result_nodes = []
        for i, task in enumerate(tasks):
            # Divider between tasks, but not after the last one — a document/section
            # may not end with a transition.
            add_divider = i < len(tasks) - 1
            result_nodes.extend(
                self._build_task_section(task, images_dir, details, add_divider=add_divider)
            )

        return result_nodes

    def _build_task_section(self, task, images_dir, details, add_divider=True):
        """Build docutils nodes for a single task entry."""
        section_nodes = []

        display_name = task["name"].replace("_", " ").capitalize()

        # Section with title (renders as a proper heading)
        section_id = f"task-{task['name']}"
        section = nodes.section(ids=[section_id])
        section += nodes.title(text=display_name)
        section_nodes.append(section)

        # Images (PNG) — Sphinx copies these to _images/ automatically.
        image_paths = _find_task_images(task["name"], images_dir)
        for img_path in image_paths:
            rel_path = os.path.relpath(img_path, self.env.srcdir)
            rel_path = rel_path.replace("\\", "/")
            img_node = nodes.image(uri=rel_path, width="600px")
            section += img_node

        # Videos (MP4) — rendered as an inline HTML5 <video> player. The files are
        # copied into the HTML build's _images/ directory by the build-finished
        # handler (_copy_task_videos) registered in setup().
        video_paths = _find_task_videos(task["name"], images_dir)
        for vid_path in video_paths:
            self.env.note_dependency(str(vid_path))
            fname = os.path.basename(vid_path)
            html = (
                '<video controls preload="metadata" width="600" '
                'style="max-width:100%; height:auto;">'
                f'<source src="_images/{fname}" type="video/mp4">'
                'Your browser does not support the video tag.'
                '</video>'
            )
            section += nodes.raw("", html, format="html")

        # Field list for task info
        field_list = nodes.field_list()
        task_detail = details.get(task["name"])

        # All content fields come from JSON (task_details.json)
        # 1. Summary
        desc_text = task_detail.get("short_description", "") if task_detail else ""
        if desc_text:
            field = nodes.field()
            field += nodes.field_name(text="Summary")
            field_body = nodes.field_body()
            field_body += nodes.paragraph(text=desc_text)
            field += field_body
            field_list += field

        # 2. Details
        detailed_desc = task_detail.get("detailed_description", "") if task_detail else ""
        if detailed_desc:
            field = nodes.field()
            field += nodes.field_name(text="Details")
            field_body = nodes.field_body()
            field_body += nodes.paragraph(text=detailed_desc)
            field += field_body
            field_list += field

        # 3. Recorded metrics
        metrics_text = task_detail.get("recorded_metrics", "") if task_detail else ""
        if metrics_text:
            field = nodes.field()
            field += nodes.field_name(text="Recorded metrics")
            field_body = nodes.field_body()
            field_body += nodes.paragraph(text=metrics_text)
            field += field_body
            field_list += field

        # 4. Conditions
        conditions_raw = task_detail.get("conditions", "") if task_detail else ""
        if conditions_raw:
            conds = [c.strip() for c in conditions_raw.split(",") if c.strip()]
            cond_text = ", ".join(conds)
            field = nodes.field()
            field += nodes.field_name(text="Conditions")
            field_body = nodes.field_body()
            field_body += nodes.paragraph(text=cond_text)
            field += field_body
            field_list += field

        # 5. Reference(s). Accept either a single string or a list of strings;
        # each citation is rendered as its own paragraph.
        ref_value = task_detail.get("reference", "") if task_detail else ""
        refs = ref_value if isinstance(ref_value, list) else ([ref_value] if ref_value else [])
        refs = [r for r in refs if r]
        if refs:
            field = nodes.field()
            field += nodes.field_name(text="References" if len(refs) > 1 else "Reference")
            field_body = nodes.field_body()
            for r in refs:
                field_body += nodes.paragraph(text=r)
            field += field_body
            field_list += field

        if len(field_list) > 0:
            section += field_list

        # 6. Task file columns dropdown (from JSON). Documents the task-specific
        # columns of the generated task TSV. The general/shared columns (trial_num,
        # trial_dur, stim, hand, condition, etc.) are documented once on the Getting
        # Started page, so they are not repeated per task here.
        #
        # Prefer the newer "task_file_columns" key (Column / Type / Description). If a
        # task has not been converted yet, fall back to the older "task_file_parameters"
        # key (which documented the make_task_file() arguments — now in the API docs).
        if task_detail:
            columns = task_detail.get("task_file_columns", {})
            if columns:
                rst_lines = [
                    ".. dropdown:: Task file columns",
                    "",
                    "   .. list-table::",
                    "      :header-rows: 1",
                    "      :widths: 20 12 68",
                    "",
                    "      * - Column",
                    "        - Type",
                    "        - Description",
                ]
                for col_name, col_info in columns.items():
                    ctype = col_info.get("type", "")
                    desc = col_info.get("description", "")
                    rst_lines.append(f"      * - ``{col_name}``")
                    rst_lines.append(f"        - {ctype}")
                    rst_lines.append(f"        - {desc}")

                rst_lines.append("")
                string_list = StringList(rst_lines)
                wrapper = nodes.container()
                self.state.nested_parse(string_list, 0, wrapper)
                section += wrapper
            else:
                params = task_detail.get("task_file_parameters", {})
                if params:
                    rst_lines = [
                        ".. dropdown:: Task file parameters",
                        "",
                        "   .. list-table::",
                        "      :header-rows: 1",
                        "      :widths: 15 10 10 65",
                        "",
                        "      * - Parameter",
                        "        - Type",
                        "        - Default",
                        "        - Description",
                    ]
                    for param_name, param_info in params.items():
                        ptype = param_info.get("type", "")
                        default = param_info.get("default", "")
                        desc = param_info.get("description", "")
                        rst_lines.append(f"      * - ``{param_name}``")
                        rst_lines.append(f"        - {ptype}")
                        rst_lines.append(f"        - {default}")
                        rst_lines.append(f"        - {desc}")

                    rst_lines.append("")
                    string_list = StringList(rst_lines)
                    wrapper = nodes.container()
                    self.state.nested_parse(string_list, 0, wrapper)
                    section += wrapper

        # Horizontal divider between tasks (omitted after the last task, since a
        # section/document may not end with a transition).
        if add_divider:
            section += nodes.transition()

        return section_nodes


class TaskSummaryTableDirective(SphinxDirective):
    """Directive that generates a summary table of tasks from task_table.tsv."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        source_dir = Path(self.env.srcdir)
        tsv_path = source_dir.parent / "MultiTaskBattery" / "task_table.tsv"

        self.env.note_dependency(str(tsv_path))

        tasks = _read_task_table(tsv_path)
        tasks.sort(key=lambda t: t["name"])

        # Build table
        table = nodes.table()
        tgroup = nodes.tgroup(cols=3)
        table += tgroup

        for width in [30, 20, 50]:
            tgroup += nodes.colspec(colwidth=width)

        # Header
        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        for header_text in ["Name", "Code", "Descriptive Name"]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=header_text)
            header_row += entry
        thead += header_row

        # Body
        tbody = nodes.tbody()
        tgroup += tbody
        for task in tasks:
            row = nodes.row()
            for field in ["name", "code", "descriptive_name"]:
                entry = nodes.entry()
                text = task.get(field, "").strip().strip('"')
                entry += nodes.paragraph(text=text)
                row += entry
            tbody += row

        return [table]


def _copy_task_videos(app, exception):
    """Copy task demo MP4s from images/ into the HTML build's _images/ directory.

    Sphinx auto-copies files referenced by image nodes, but our videos are emitted
    as raw HTML (<video src="_images/...">), so we copy them here. Only runs for the
    HTML builder; other builders (e.g. latexpdf) ignore the raw-html video nodes.
    """
    if exception is not None:
        return
    builder = getattr(app, "builder", None)
    if builder is None or getattr(builder, "format", None) != "html":
        return

    images_dir = Path(app.srcdir) / "images"
    if not images_dir.is_dir():
        return

    out_images = Path(app.outdir) / "_images"
    out_images.mkdir(parents=True, exist_ok=True)
    for mp4 in images_dir.glob("*.mp4"):
        try:
            shutil.copy2(str(mp4), str(out_images / mp4.name))
        except OSError:
            pass


def setup(app):
    """Register the extension with Sphinx."""
    app.add_directive("task-descriptions", TaskDescriptionsDirective)
    app.add_directive("task-summary-table", TaskSummaryTableDirective)
    app.connect("build-finished", _copy_task_videos)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
