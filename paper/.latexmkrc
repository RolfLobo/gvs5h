# Serialise the engine so two builds cannot write the PDF at the same time, and record each
# invocation so it is visible which builds go through this file.
#
# Build-on-change fires a fresh latexmk while the previous one is still running. Both write
# the same PDF, and the result is a file of exactly the right length whose xref table
# describes the other run's body: it opens, and every page is blank.
#
# No percent signs below: latexmk substitutes its own placeholders (%T is the tex file, %O
# the options, %S the source) before the shell ever sees the string.
my $lock = "$ENV{HOME}/.cache/gvs5h-paper.lock";
my $log  = "$ENV{HOME}/.cache/gvs5h-build.log";
$pdflatex = "sh -c 'echo \"\$(date -Ins) \$\$ \$PWD\" >> $log; exec flock $lock pdflatex \"\$\@\"' sh %O %S";
