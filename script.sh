pyinstaller -F -n wine --windowed -p . \
--hidden-import="sklearn.metrics._pairwise_distances_reduction._datasets_pair" \
--hidden-import="sklearn.metrics._pairwise_distances_reduction._middle_term_computer" \
--hidden-import="PIL._tkinter_finder" \
--hidden-import="scipy.fftpack" \
shit.py