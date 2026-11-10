# Print the CHANGELOG.md section for one version as GitHub release notes.
#
#   awk -v version=0.1.2 -f packaging/release_notes.awk CHANGELOG.md
#
# The section runs from "## <version> ..." to the next "## " heading. GitHub renders every
# newline in release notes as a line break, so indented continuation lines are joined onto
# the line they continue.

/^## / {
    if (found) exit
    found = ($2 == version)
    next
}

!found { next }

{
    if (have && buf !~ /^[ \t]*$/ && $0 ~ /^[ \t]+[^ \t]/) {
        sub(/^[ \t]+/, "")
        buf = buf " " $0
        next
    }
    if (have) print buf
    buf = $0
    have = 1
}

END { if (have) print buf }
