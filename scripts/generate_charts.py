import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# --- Chart 1: Total MIPS Comparison (Basicmath) ---
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

kernels = ['Basicmath', 'FFT', 'Matmul', 'Qsort', 'Stringsearch']
x = np.arange(len(kernels))
width = 0.12

four_2 = [116.0, 135.1, 136.0, 93.0, 98.5]
four_4 = [233.0, 263.9, 259.6, 159.4, 257.6]
four_8 = [420.7, 496.8, 488.3, 214.0, 356.9]
six_2 = [49.0, 60.8, 60.8, 33.5, 75.2]
six_4 = [86.8, 116.5, 106.2, 59.8, 169.4]
six_8 = [127.2, 214.7, 181.3, 105.5, 319.7]

bars1 = ax.bar(x - 2.5*width, four_2, width, label='4-stage 2C', color='#00b4d8')
bars2 = ax.bar(x - 1.5*width, four_4, width, label='4-stage 4C', color='#0077b6')
bars3 = ax.bar(x - 0.5*width, four_8, width, label='4-stage 8C', color='#023e8a')
bars4 = ax.bar(x + 0.5*width, six_2, width, label='6-stage 2C', color='#f4845f')
bars5 = ax.bar(x + 1.5*width, six_4, width, label='6-stage 4C', color='#e63946')
bars6 = ax.bar(x + 2.5*width, six_8, width, label='6-stage 8C', color='#9d0208')

ax.set_ylabel('Total MIPS', fontsize=13, color='white', fontweight='bold')
ax.set_title('Total MIPS Comparison Across Kernels (Higher is Better)', fontsize=15, color='white', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(kernels, fontsize=11, color='white')
ax.tick_params(axis='y', colors='white', labelsize=10)
ax.legend(loc='upper left', fontsize=9, facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white', ncol=2)
ax.grid(axis='y', alpha=0.2, color='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart_mips.png'), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
plt.close()
print("chart_mips.png saved")

# --- Chart 2: Scaling Efficiency ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

cores = [2, 4, 8]
eff_4_basic = [1.000, 0.974, 0.834]
eff_4_fft = [1.000, 0.935, 0.431]
eff_4_matmul = [1.000, 3.728, 3.397]
eff_6_basic = [1.000, 0.785, 0.385]
eff_6_fft = [1.000, 0.698, 0.243]

ax.plot(cores, eff_4_basic, 'o-', color='#00b4d8', linewidth=2.5, markersize=8, label='4-stage Basicmath')
ax.plot(cores, eff_4_fft, 's-', color='#0077b6', linewidth=2.5, markersize=8, label='4-stage FFT')
ax.plot(cores, eff_6_basic, 'o--', color='#f4845f', linewidth=2.5, markersize=8, label='6-stage Basicmath')
ax.plot(cores, eff_6_fft, 's--', color='#e63946', linewidth=2.5, markersize=8, label='6-stage FFT')

ax.set_xlabel('Number of Cores', fontsize=13, color='white', fontweight='bold')
ax.set_ylabel('Efficiency', fontsize=13, color='white', fontweight='bold')
ax.set_title('Scaling Efficiency (Relative to 2-Core Baseline)', fontsize=14, color='white', fontweight='bold', pad=15)
ax.set_xticks(cores)
ax.tick_params(colors='white', labelsize=11)
ax.legend(fontsize=10, facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white')
ax.grid(alpha=0.2, color='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart_efficiency.png'), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
plt.close()
print("chart_efficiency.png saved")

# --- Chart 3: FPGA LUT Utilization ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

cores_labels = ['2 Cores', '4 Cores', '8 Cores']
x = np.arange(len(cores_labels))
width = 0.35

lut_4 = [5921, 11048, 28591]
lut_6 = [11304, 22155, 43731]

bars1 = ax.bar(x - width/2, lut_4, width, label='4-stage Pipeline', color='#00b4d8', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, lut_6, width, label='6-stage Multicycle', color='#e63946', edgecolor='white', linewidth=0.5)

for bar, val in zip(bars1, lut_4):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500, f'{val:,}', ha='center', va='bottom', fontsize=10, color='#00b4d8', fontweight='bold')
for bar, val in zip(bars2, lut_6):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500, f'{val:,}', ha='center', va='bottom', fontsize=10, color='#e63946', fontweight='bold')

ax.set_ylabel('LUT Count', fontsize=13, color='white', fontweight='bold')
ax.set_title('FPGA Resource Utilization - LUTs (Zynq-7020)', fontsize=14, color='white', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cores_labels, fontsize=12, color='white')
ax.tick_params(axis='y', colors='white', labelsize=10)
ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white')
ax.set_ylim(0, 52000)
ax.grid(axis='y', alpha=0.2, color='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart_lut.png'), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
plt.close()
print("chart_lut.png saved")

# --- Chart 4: Power Consumption ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

power_4 = [1.446, 1.475, 1.63]
power_6 = [1.559, 1.605, 1.69]

ax.bar(x - width/2, power_4, width, label='4-stage Pipeline', color='#00b4d8', edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, power_6, width, label='6-stage Multicycle', color='#e63946', edgecolor='white', linewidth=0.5)

for i, (p4, p6) in enumerate(zip(power_4, power_6)):
    ax.text(i - width/2, p4 + 0.02, f'{p4}W', ha='center', va='bottom', fontsize=11, color='#00b4d8', fontweight='bold')
    ax.text(i + width/2, p6 + 0.02, f'{p6}W', ha='center', va='bottom', fontsize=11, color='#e63946', fontweight='bold')

ax.set_ylabel('Total Power (W)', fontsize=13, color='white', fontweight='bold')
ax.set_title('Total Power Consumption (Zynq-7020)', fontsize=14, color='white', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cores_labels, fontsize=12, color='white')
ax.tick_params(axis='y', colors='white', labelsize=10)
ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white')
ax.set_ylim(1.3, 1.85)
ax.grid(axis='y', alpha=0.2, color='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart_power.png'), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
plt.close()
print("chart_power.png saved")

# --- Chart 5: Speedup comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

kernels_sp = ['Basicmath', 'FFT', 'Matmul', 'Qsort', 'Stringsearch']
speedup_4_8 = [3.336, 1.724, 13.589, 1.103, 2.148]
speedup_6_8 = [1.541, 0.973, 2.447, 0.308, 0.734]

x_sp = np.arange(len(kernels_sp))
ax.bar(x_sp - width/2, speedup_4_8, width, label='4-stage (8-core)', color='#00b4d8', edgecolor='white', linewidth=0.5)
ax.bar(x_sp + width/2, speedup_6_8, width, label='6-stage (8-core)', color='#e63946', edgecolor='white', linewidth=0.5)

ax.axhline(y=1.0, color='#ffd60a', linewidth=1, linestyle='--', alpha=0.7, label='Baseline (2-core)')

ax.set_ylabel('Speedup (vs 2-core)', fontsize=13, color='white', fontweight='bold')
ax.set_title('8-Core Speedup Relative to 2-Core Baseline', fontsize=14, color='white', fontweight='bold', pad=15)
ax.set_xticks(x_sp)
ax.set_xticklabels(kernels_sp, fontsize=11, color='white')
ax.tick_params(axis='y', colors='white', labelsize=10)
ax.legend(fontsize=10, facecolor='#16213e', edgecolor='#e0e0e0', labelcolor='white')
ax.grid(axis='y', alpha=0.2, color='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart_speedup.png'), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
plt.close()
print("chart_speedup.png saved")

print("\nAll charts generated successfully!")
