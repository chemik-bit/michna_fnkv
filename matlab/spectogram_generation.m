close all, clc, clear all
curr_dir = pwd;
curr_disk = split(curr_dir,":")
string(curr_disk(1))+":**/fnkv_voices/data/svdadult/svdadult_renamed"

%%
close all, clc, clear all
file_name = "svdadult0101_unhealthy_50000.txt"
old_data = readmatrix(file_name);
old_sample_rate = 50000;
new_sample_rate = 50000;
window_len = 0.025;
overlap = 125;
data_len = 1;
data = resample(old_data, new_sample_rate, old_sample_rate);
center = fix(length(data)/2);
start = fix(center - data_len / 2 * new_sample_rate);
stop = fix(center + data_len / 2 * new_sample_rate);
data = data(start:stop);

fig = figure("units", "pixels", "Position", [0 500 399 26]);
ax = axes(fig);
ax.Position = [0 0 1 1];
spectrogram(data, fix(window_len*new_sample_rate), overlap, "power", "xaxis");
colorbar(ax, "off")
set(ax, "xtick", [])
set(ax, "ytick", [])
set(fig, 'MenuBar', 'none');
set(fig, 'ToolBar', 'none');
colormap gray