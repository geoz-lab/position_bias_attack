% =============================================================================
%  Paste-ready LaTeX for the paper (intended as §7 / an appendix section).
%  Preamble needs: \usepackage{booktabs}, \usepackage{tabularx},
%                  \usepackage[most]{tcolorbox}  (only for the prompt boxes).
%  Data source: position_bias_check/results/prompt_check.csv (raw JSON in results/raw/).
%  Setup: causal reranker (Llama-3.2-3B LFT adapter `lft_movielens`), MovieLens
%         anchor, N=300 candidate sets, attacker budget R=50, seed 42.
% =============================================================================

\newtcolorbox{promptbox}[1]{colback=gray!4,colframe=black!55,fonttitle=\bfseries,
  title=#1,boxrule=0.5pt,left=3pt,right=3pt,top=3pt,bottom=3pt,before skip=6pt,after skip=6pt}

\section{Prompt-Level Mitigation Does Not Close the Attack Surface}
\label{sec:prompt-mitigation}

A natural first reaction to the position-bias attack is to simply \emph{instruct} the
reranker to ignore candidate order. We test whether this works. Holding the \emph{same}
trained causal reranker fixed (same weights, causal attention, standard RoPE), we change
\emph{only} the eval-time instruction and re-run both probes---the position scan
(\textsc{curve\_range}) and the budget-$R$ permutation attacker (\textsc{promo@5}).
If a prompt genuinely removes the bias, its metrics should collapse toward $0$, matching
the architectural / consistency-trained defenses; if they stay at the baseline, the bias
is \emph{mechanistic} rather than \emph{instructional}.

All prompts share the listwise template below; only the \textsc{Instruction} slot varies.
Candidate and history formatting are byte-identical across variants, so wording is the
sole moving part.

\begin{promptbox}{Shared listwise template (InvariRank-style)}
\ttfamily\small
[SPAN]\\
\textit{$\langle$Instruction$\rangle$}\\
User history:\\
title: $\ldots$\\
[/SPAN]\\
\ [ITEM] $\langle$candidate 1$\rangle$ [/ITEM]\\
\ [ITEM] $\langle$candidate 2$\rangle$ [/ITEM]\\
\ $\ldots$
\end{promptbox}

We compare a control ($p_0$, the original wording) against five treatments that each
encode the ``ignore order / treat as a set'' intent in a \emph{different prompting style},
so a null result cannot be blamed on one unlucky phrasing.

\begin{table}[t]
\centering\small
\caption{Instruction wordings tested (the $\langle$Instruction$\rangle$ slot).}
\label{tab:prompt-wordings}
\begin{tabularx}{\linewidth}{@{}llX@{}}
\toprule
ID & Style & Instruction \\
\midrule
$p_0$ & control & Given the user's interaction history, rank the candidate items according to the user's preferences. \\
$p_1$ & declarative & \ldots rank the candidate items \ldots The candidates are provided together as an \textbf{unordered set}: consider all of them as a group and judge each item only on its own merits, not on where it appears in the list. \\
$p_2$ & imperative & Rank the candidate items by relevance \ldots \textbf{Ignore the list order. Do not favor earlier or later items.} The position of a candidate is random and carries no meaning: judge content only. \\
$p_3$ & persona & \textbf{You are a position-invariant recommendation reranker.} Your output depends only on item content and the user's history, never on the order in which candidates are fed to you \ldots \\
$p_4$ & chain-of-thought & \ldots \textbf{Reason step by step:} assess each candidate on its own, then compare those per-item assessments \ldots the ranking must be the same no matter how the candidates were ordered. \\
$p_5$ & structured rules & Rules: (1) treat the candidates as an unordered set; (2) the listed order is random and must be ignored; (3) score by relevance only; (4) \textbf{an item's rank must not change if the list is reshuffled.} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\centering\small
\caption{Prompting does not reduce the attack surface. Causal reranker, MovieLens anchor,
$N{=}300$, $R{=}50$. \textsc{curve\_range} and \textsc{promo@5} are on the irrelevant
stratum; larger $=$ more vulnerable. For reference, the architectural-invariance defense (A)
reaches \textsc{promo@5}$=0.000$ (\textsc{curve\_range}$=0.06$) and the consistency-KL
defense (B2) reaches \textsc{promo@5}$=0.007$ on the same anchor.}
\label{tab:prompt-results}
\begin{tabular}{@{}llcccc@{}}
\toprule
ID & Style & curve\_range & promo@5 & rank\_gain & into\_top5 \\
\midrule
$p_0$ & control (baseline)   & 3.09 & 0.120 & 3.34 & 0.273 \\
$p_1$ & declarative          & 3.12 & 0.120 & 3.48 & 0.277 \\
$p_2$ & imperative           & 3.15 & 0.123 & 3.48 & 0.280 \\
$p_3$ & persona              & 2.93 & 0.127 & 3.14 & 0.273 \\
$p_4$ & chain-of-thought     & 3.20 & 0.123 & 3.53 & 0.277 \\
$p_5$ & structured rules     & 3.03 & 0.130 & 3.40 & 0.280 \\
\midrule
\multicolumn{2}{@{}l}{\textit{architectural invariance} (A)} & 0.06 & \textbf{0.000} & --- & --- \\
\multicolumn{2}{@{}l}{\textit{consistency-KL} (B2)}          & ---  & \textbf{0.007} & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Finding.} Prompting is inert. Across all five styles, \textsc{promo@5} stays in
$[0.120, 0.130]$ and \textsc{curve\_range} in $[2.93, 3.20]$---statistically indistinguishable
from the control (within the $\pm 0.026$ seed margin of the main study), and if anything the
``stronger'' anti-position wordings ($p_5$, $p_3$) are marginally \emph{worse}. This stands in
sharp contrast to the architectural (A) and training-time (B2) defenses, which drive
\textsc{promo@5} to $\approx 0$. We conclude that the position bias is \textbf{mechanistic}
(a property of causal attention $+$ RoPE), not \textbf{instructional}: it cannot be prompted
away and requires an architectural or training-time intervention. As a sanity check, the
control $p_0$ exactly reproduces the known anchor numbers
(\textsc{promo@5}$=0.120$, \textsc{curve\_range}$=3.09$, \textsc{rank\_gain}$=3.34$,
\textsc{into\_top5}$=0.273$), confirming the harness is faithful.
