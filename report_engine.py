from datetime import date
import time
import git

class Report:

    def __init__(self):

        repo = git.Repo(search_parent_directories=True)

        self.body = '<h1>Multicomponent condensation run report</h1>'

        report_header_paragraph_lines = [
            f'Date: {date.today()}',
            f'Time: {time.strftime("%H:%M:%S")}',
            f'Git hash: {repo.head.object.hexsha}',
            f'Git branch: {repo.head.ref}',
            f'Git directory status: {"dirty" if repo.is_dirty(untracked_files=True) else "clean"}'
        ]

        self.body += '<p>' + '<br>'.join(report_header_paragraph_lines) + '</p>'
        self.body += '<hr/>'

    def add_section_header(self, text: str):
        self.body += f'<h2>{text}</h1>'

    def add_section_paragraph(self, lines: list[str]):
        self.body += f'<p>{"<br/>".join(lines)}</p>'

    def save_report(self, path):
        html = f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Report</title></head><body>{self.body}</body></html>'
        with open(path, 'w') as file:
            file.write(html)


if __name__ == '__main__':
    report = Report()
    report.save_report('test.html')
