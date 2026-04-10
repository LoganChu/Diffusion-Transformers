#!/bin/bash

FILE="report.tex"

echo "LaTeX Environment Balance Check:"
echo "================================"

begin_doc=$(grep -c "\\begin{document}" "$FILE")
end_doc=$(grep -c "\\end{document}" "$FILE")
echo "  document: $begin_doc begin, $end_doc end"

begin_fig=$(grep -c "\\begin{figure}" "$FILE")
end_fig=$(grep -c "\\end{figure}" "$FILE")
echo "  figure: $begin_fig begin, $end_fig end"

begin_tikz=$(grep -c "\\begin{tikzpicture}" "$FILE")
end_tikz=$(grep -c "\\end{tikzpicture}" "$FILE")
echo "  tikzpicture: $begin_tikz begin, $end_tikz end"

begin_table=$(grep -c "\\begin{table}" "$FILE")
end_table=$(grep -c "\\end{table}" "$FILE")
echo "  table: $begin_table begin, $end_table end"

echo ""
echo "Content Analysis:"
echo "================"
sections=$(grep -c "^\\section{" "$FILE")
echo "  Sections: $sections"

cite_count=$(grep -c "\\cite{" "$FILE")
echo "  Citations: $cite_count"

speedups=$(grep -c "1.81\|1.51\|2.19\|5.01\|1.23" "$FILE")
echo "  Speedup references: $speedups"

echo ""
echo "Critical Checks:"
echo "==============="
if grep -q "\\usepackage{proposal}" "$FILE"; then
  echo "  ✓ proposal.sty imported"
else
  echo "  ✗ proposal.sty NOT imported"
fi

if grep -q "\\begin{figure}" "$FILE" && grep -q "\\end{figure}" "$FILE"; then
  echo "  ✓ Figure environment present and balanced"
else
  echo "  ✗ Figure environment error"
fi

if grep -q "\\caption{End-to-end" "$FILE"; then
  echo "  ✓ Figure caption present"
else
  echo "  ✗ Figure caption missing"
fi

echo ""
echo "File Info:"
echo "=========="
wc -l "$FILE"

