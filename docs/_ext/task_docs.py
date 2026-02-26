"""
Sphinx extension to auto-generate task descriptions from task_table.tsv. This way we dont have to manually add tasks.
The only thing that can be added manually (optional) is images for each task. If yest just go to images/ and add {task_name}.png, {task_name}_2.png, etc.

Usage in an .rst file:

    .. task-descriptions::

    .. task-summary-table::
"""

import csv
import os
from pathlib import Path
from docutils import nodes
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


def _find_task_images(task_name, images_dir):
    """Find images for a task: {name}.png, {name}_2.png, {name}_3.png, ..."""
    images = []
    primary = images_dir / f"{task_name}.png"
    if primary.exists():
        images.append(primary)

    n = 2
    while True:
        variant = images_dir / f"{task_name}_{n}.png"
        if variant.exists():
            images.append(variant)
            n += 1
        else:
            break

    return images


class TaskDescriptionsDirective(SphinxDirective):
    """Directive that generates task description sections from task_table.tsv."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        source_dir = Path(self.env.srcdir)
        tsv_path = source_dir.parent / "MultiTaskBattery" / "task_table.tsv"
        images_dir = source_dir / "images"

        self.env.note_dependency(str(tsv_path))

        tasks = _read_task_table(tsv_path)
        tasks.sort(key=lambda t: t["name"])

        result_nodes = []
        for task in tasks:
            result_nodes.extend(self._build_task_section(task, images_dir))

        return result_nodes

    def _build_task_section(self, task, images_dir):
        """Build docutils nodes for a single task entry."""
        section_nodes = []

        display_name = task["name"].replace("_", " ").capitalize()

        # Section with title (renders as a proper heading)
        section_id = f"task-{task['name']}"
        section = nodes.section(ids=[section_id])
        section += nodes.title(text=display_name)
        section_nodes.append(section)

        # Images
        image_paths = _find_task_images(task["name"], images_dir)
        for img_path in image_paths:
            rel_path = os.path.relpath(img_path, self.env.srcdir)
            rel_path = rel_path.replace("\\", "/")
            img_node = nodes.image(uri=rel_path, width="600px")
            section += img_node

        # Description
        desc_text = task.get("description", "").strip().strip('"')
        if desc_text and desc_text.upper() != "NA":
            desc_para = nodes.paragraph(text=desc_text)
            section += desc_para

        # Conditions
        conditions_raw = task.get("conditions", "").strip().strip('"')
        if conditions_raw and conditions_raw.upper() != "NA":
            cond_title = nodes.paragraph(text="Available conditions:")
            cond_list = nodes.bullet_list()
            for cond in conditions_raw.split(","):
                cond = cond.strip()
                if cond:
                    item = nodes.list_item()
                    item += nodes.paragraph(text=cond)
                    cond_list += item
            section += cond_title
            section += cond_list

        # Reference
        ref_text = task.get("reference", "").strip().strip('"')
        if ref_text and ref_text.upper() != "NA":
            ref_para = nodes.paragraph()
            ref_para += nodes.strong(text="Reference: ")
            ref_para += nodes.Text(ref_text)
            section += ref_para

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

        for width in [20, 15, 65]:
            tgroup += nodes.colspec(colwidth=width)

        # Header
        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        for header_text in ["Name", "Code", "Description"]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=header_text)
            header_row += entry
        thead += header_row

        # Body
        tbody = nodes.tbody()
        tgroup += tbody
        for task in tasks:
            row = nodes.row()
            for field in ["name", "code", "description"]:
                entry = nodes.entry()
                text = task.get(field, "").strip().strip('"')
                entry += nodes.paragraph(text=text)
                row += entry
            tbody += row

        return [table]


def setup(app):
    """Register the extension with Sphinx."""
    app.add_directive("task-descriptions", TaskDescriptionsDirective)
    app.add_directive("task-summary-table", TaskSummaryTableDirective)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
