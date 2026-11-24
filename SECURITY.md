# Security Policy

## Supported versions

This project is in active pre-1.0 development. Security fixes are applied to the
latest release and the `main` branch only.

| Version        | Supported |
| -------------- | --------- |
| latest release | ✅        |
| older releases | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately through GitHub's
[private vulnerability reporting](https://github.com/tempoo04/image-inspector/security/advisories/new):
go to the repository's **Security** tab → **Report a vulnerability**.

When reporting, please include:

- a description of the vulnerability and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected version or commit, and
- any suggested remediation.

You can expect an initial acknowledgement within a few days. Once a fix is
ready, a release will be published and the advisory disclosed.

## Scope notes

This is a **local-first** desktop/web tool: it reads image files from paths the
user chooses and stores metadata in a local SQLite database. It does not upload
data or expose a network service by default. Reports most relevant to this
project include: path-traversal or unsafe file handling during scanning or
export, unsafe deserialization, and vulnerabilities in how untrusted image
files are parsed.
