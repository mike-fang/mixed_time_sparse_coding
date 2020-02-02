let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +43 ~/python_projects/mixed_time_sparse_coding/bars.py
badd +59 ~/python_projects/mixed_time_sparse_coding/bars_learn_pi.py
badd +54 ~/python_projects/mixed_time_sparse_coding/no_norm_A.py
badd +57 ~/python_projects/mixed_time_sparse_coding/soln_analysis.py
badd +107 ~/python_projects/mixed_time_sparse_coding/ctsc.py
badd +4 ~/python_projects/mixed_time_sparse_coding/scratch_pad.py
badd +11 ~/python_projects/mixed_time_sparse_coding/learn_pi.py
badd +93 ~/python_projects/mixed_time_sparse_coding/loaders.py
badd +102 term://~/python_projects/mixed_time_sparse_coding//17962:python\ learn_pi.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//17974:python\ learn_pi.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//17991:python\ learn_pi.py
badd +102 term://~/python_projects/mixed_time_sparse_coding//17998:python\ learn_pi.py
badd +1 ~/python_projects/mixed_time_sparse_coding/bars_likelihood.py
badd +12 term://~/python_projects/mixed_time_sparse_coding//19042:python\ learn_pi.py
badd +43 ~/python_projects/mixed_time_sparse_coding/l0_sparse_prior.py
badd +2 ~/python_projects/mixed_time_sparse_coding/euler_maruyama.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//20946:python\ l0_sparse_prior.py
badd +92 ~/python_projects/mixed_time_sparse_coding/lca.py
badd +36 ~/python_projects/mixed_time_sparse_coding/lca_bars.py
argglobal
%argdel
$argadd bars.py
edit ~/python_projects/mixed_time_sparse_coding/bars_learn_pi.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 68 + 102) / 204)
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 65 - ((36 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
65
normal! 090|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/bars.py") | buffer ~/python_projects/mixed_time_sparse_coding/bars.py | else | edit ~/python_projects/mixed_time_sparse_coding/bars.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/bars.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 31 - ((7 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
31
normal! 010|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/no_norm_A.py") | buffer ~/python_projects/mixed_time_sparse_coding/no_norm_A.py | else | edit ~/python_projects/mixed_time_sparse_coding/no_norm_A.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/no_norm_A.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
20
normal! zo
let s:l = 89 - ((47 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
89
normal! 038|
wincmd w
3wincmd w
exe 'vert 1resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 68 + 102) / 204)
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOF
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
