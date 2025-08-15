import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import RectangleSelector, Button, TextBox
import pandas as pd
import os
from scipy.signal import find_peaks
from numpy.polynomial.chebyshev import Chebyshev
from matplotlib.widgets import RadioButtons


def toggle_background(event):
    global background_enabled
    background_enabled = not background_enabled
    if background_enabled == False:
        update_fit_plot(current_index)
        background_button.label.set_text('Bkg Off')
    else:
        background_button.label.set_text('Bkg On')
        update_fit_plot(current_index)

def find_non_peak_regions(x, y, threshold, bkg_peak_prominence, bkg_peak_width_multiplier, peak_distance):
    """
    Identify the indices in x and y that are not part of peaks.
    
    Parameters:
        x (array-like): x values of the data
        y (array-like): y values of the data
        threshold (float): Minimum height required to qualify as a peak
        peak_distance (int): Minimum distance between detected peaks
        
    Returns:
        non_peak_indices (array-like): Boolean mask of non-peak regions
    """
    peaks, _ = find_peaks(y, height=threshold, prominence=bkg_peak_prominence, distance=peak_distance)
    non_peak_mask = np.ones_like(y, dtype=bool)
    non_peak_mask[peaks] = False
    for peak in peaks:
        non_peak_mask[max(0, peak - int(bkg_peak_width_multiplier * peak_distance)):min(len(y), peak + int(bkg_peak_width_multiplier * peak_distance))] = False
    return non_peak_mask

def simple_linear(x, m, x0, y0):
    y = y0 + m * (x - x0)
    return y

def background_subtraction(x, y, poly_order=6, threshold=5, bkg_peak_prominence=5, bkg_peak_width_multiplier=1, peak_distance=25):
    """
    Perform background subtraction by fitting a Chebyshev polynomial to non-peak data.
    
    Parameters:
        x (array-like): x values of the data
        y (array-like): y values of the data
        poly_order (int): Order of the Chebyshev polynomial
        threshold (float): Threshold for peak detection in find_peaks
        peak_distance (int): Minimum distance between detected peaks
        
    Returns:
        y_corrected (array-like): Background-corrected y data
        background (array-like): Fitted background polynomial values
    """
    non_peak_mask = find_non_peak_regions(x, y, threshold, bkg_peak_prominence, bkg_peak_width_multiplier, peak_distance)
    y_bkg = y.copy()
    idxs = np.array([])
    x_values = np.array([])
    y_values = np.array([])
    for i, value in enumerate(non_peak_mask[:-1]):
        if non_peak_mask[i + 1] == False or value == False:
            idxs = np.append(idxs, i)
            x_values = np.append(x_values, x[i])
            y_values = np.append(y_values, y[i])
        elif i > 0:
            if value != False and non_peak_mask[i - 1] == False:
                idxs = np.append(idxs, i)
                x_values = np.append(x_values, x[i])
                y_values = np.append(y_values, y[i])
                idxs = idxs.astype(int)
                xmin = x_values[0]
                xmax = x_values[-1]
                ymin = y_values[0]
                ymax = y_values[-1]
                m = (ymax - ymin) / (xmax - xmin)
                pseudo_y_bkg = simple_linear(x_values, m, xmin, ymin)
                np.put(y_bkg, idxs, pseudo_y_bkg)
                idxs = np.array([])
                x_values = np.array([])
                y_values = np.array([])
        else:
            pass
    s = pd.Series(y_bkg)
    s = s.rolling(5, min_periods=1).mean()
    y_bkg = np.array(s)
    bkg_non_peaks_x = x[non_peak_mask]
    bkg_non_peaks_y = y_bkg[non_peak_mask]
    peak_mask = np.logical_not(non_peak_mask)
    bkg_peaks_x = x[peak_mask]
    bkg_peaks_y = y_bkg[peak_mask]
    cheb_poly = Chebyshev.fit(x, y_bkg, poly_order)
    background = cheb_poly(x)
    y_corrected = y - background
    return (y_corrected, background, bkg_non_peaks_x, bkg_non_peaks_y, bkg_peaks_x, bkg_peaks_y)

def find_peaks_above_threshold(x, y, peak_width, threshold, distance, prominence):
    """
    Find and sort the peaks in the data by their intensity above a given threshold.

    Parameters:
    - x: array-like, the x-values (e.g., independent variable or time)
    - y: array-like, the y-values (e.g., the signal or dependent variable)
    - threshold: float, the minimum height for peaks to be considered.

    Returns:
    - peak_positions: list of x-values where peaks occur above the threshold, sorted by peak intensity.
    - peak_centers: list of x-values where peaks occur, suitable for initial guesses for the fit.
    - peak_intensities: list of peak intensities (heights), sorted by intensity.
    - num_peaks: the number of detected peaks above the threshold.
    """
    peaks, properties = find_peaks(y, height=threshold, width=peak_width, distance=distance, prominence=prominence)
    peak_positions = x[peaks]
    peak_intensities = properties['peak_heights']
    sorted_indices = np.argsort(peak_intensities)[::-1]
    peak_centers_sorted = peak_positions[sorted_indices]
    peak_intensities_sorted = peak_intensities[sorted_indices]
    num_peaks = len(peak_centers_sorted)
    return (peak_centers_sorted, peak_intensities_sorted, num_peaks)

def multi_gaussian(x, *params):
    """
    A function to calculate the sum of multiple Gaussian functions.
    """
    result = np.zeros_like(x)
    num_peaks = len(params) // 3
    for i in range(num_peaks):
        amplitude = params[3 * i]
        center = params[3 * i + 1]
        FWHM = params[3 * i + 2]
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        result += amplitude * np.exp(-0.5 * ((x - center) ** 2 / (2 * sigma ** 2)))
    return result

def pseudoVoigt_FITYK(x, amp, mu, sigma, gl):
    """Returns a 1D PseudoVoigt distribution as used in FITYK.

    Parameters:
    -----------
    x : float or array-like
        The coordinate at which to evaluate the PseudoVoigt.
    amp : float
        The peak height of the pseudovoigt.
    mu : float
        The center of the pseudovoigt.
    sigma : float
        The width parameter of the pseudovoigt.
        In this format, if function is full Gaussian or full Lorentzian, the FWHM is 2*sigma.
        Therefore any combination of the two also has a width of 2*sigma
    gl : float
        The Gaussian to Lorentzian ratio (0 for full Gaussian, 1 for full Lorentzian).
    
    Returns:
    --------
    pseudovoigt : float or array-like
        The value of the pseudovoigt function at the specified x.
    """
    gaussian = np.exp(-np.log(2) * ((x - mu) / sigma) ** 2)
    lorentzian = 1 / (1 + ((x - mu) / sigma) ** 2)
    pseudovoigt = amp * ((1 - gl) * gaussian + gl * lorentzian)
    return pseudovoigt

def multi_pseudoVoigt(x, *params):
    """
    A function to calculate the sum of multiple PseudoVoigt functions.

    Parameters:
    -----------
    x : array-like
        The coordinates at which to evaluate the sum of PseudoVoigt functions.
    *params : list
        A list of parameters, with each set of four parameters corresponding to:
        - amp (amplitude of the peak)
        - mu (center of the peak)
        - sigma (width parameter of the peak)
        - gl (Gaussian-to-Lorentzian ratio)
    
    Returns:
    --------
    result : array-like
        The summed PseudoVoigt functions evaluated at each point in x.
    """
    result = np.zeros_like(x)
    num_peaks = len(params) // 4
    for i in range(num_peaks):
        amp = params[4 * i]
        mu = params[4 * i + 1]
        FWHM = params[4 * i + 2]
        gl = params[4 * i + 3]
        sigma = FWHM / 2
        result += pseudoVoigt_FITYK(x, amp, mu, sigma, gl)
    return result

def on_select_x_range(event1, event2):
    global x_start, x_end
    x_start, x_end = (event1.xdata, event2.xdata)
    print(f'Selected x-range: ({x_start:.2f}, {x_end:.2f})')

def fit_peak_to_data(x, y, x_range, max_peaks, threshold, bkg_peak_threshold, bkg_peak_prominence, bkg_peak_width_multiplier, peak_width, peak_prominence):
    delta2th = x[1] - x[0]
    peak_distance = int(peak_width / delta2th)
    y_corrected, background, bkg_non_peaks_x, bkg_non_peaks_y, bkg_peaks_x, bkg_peaks_y = background_subtraction(x, y, poly_order=5, threshold=bkg_peak_threshold, peak_distance=peak_distance, bkg_peak_prominence=bkg_peak_prominence, bkg_peak_width_multiplier=bkg_peak_width_multiplier)
    mask = (x >= x_range[0]) & (x <= x_range[1])
    x_fit = x[mask]
    y_fit = y_corrected[mask]
    background_fit = background[mask]
    peak_centers, peak_intensities, num_peaks = find_peaks_above_threshold(x_fit, y_fit, peak_width, threshold, peak_distance, peak_prominence)
    if num_peaks == 0:
        n_peaks = 0
    elif num_peaks <= max_peaks:
        n_peaks = num_peaks
    else:
        n_peaks = max_peaks
    params_init = []
    bounds_lower = []
    bounds_upper = []
    if n_peaks == 0:
        return (x_fit, y_fit, background, background_fit, bkg_non_peaks_x, bkg_non_peaks_y, bkg_peaks_x, bkg_peaks_y, x * 0, None, None, [], y_fit * 0)
    else:
        if selected_function == 'Gaussian':
            for i in range(n_peaks):
                amp_guess = peak_intensities[i]
                cen_guess = peak_centers[i]
                wid_guess = peak_width
                gl = 0.5
                params_init.extend([amp_guess, cen_guess, wid_guess])
            params_opt, pcov = curve_fit(lambda x, *params: multi_gaussian(x, *params), x_fit, y_fit, p0=params_init)
            y_fit_curve = multi_gaussian(x_fit, *params_opt)
            y_subtracted = y_fit - y_fit_curve
            additional_peak_centers, additional_peak_intensities, additional_num_peaks = find_peaks_above_threshold(x_fit, y_subtracted, peak_width, 4 * threshold, peak_distance, 4 * peak_prominence)
            if additional_num_peaks > 0:
                peak_centers = np.append(peak_centers, additional_peak_centers)
                peak_intensities = np.append(peak_intensities, additional_peak_intensities)
                num_peaks = num_peaks + additional_num_peaks
                if num_peaks <= max_peaks:
                    n_peaks = num_peaks
                else:
                    n_peaks = max_peaks
                params_init = []
                for i in range(n_peaks):
                    amp_guess = peak_intensities[i]
                    cen_guess = peak_centers[i]
                    wid_guess = peak_width
                    gl = 0.5
                    params_init.extend([amp_guess, cen_guess, wid_guess])
                params_opt, pcov = curve_fit(lambda x, *params: multi_gaussian(x, *params), x_fit, y_fit, p0=params_init)
                y_fit_curve = multi_gaussian(x_fit, *params_opt)
                print(len(y_fit_curve))
            else:
                pass
            peak_centers = [params_opt[3 * j + 1] for j in range(n_peaks)]
        else:
            for i in range(n_peaks):
                amp_guess = peak_intensities[i]
                cen_guess = peak_centers[i]
                wid_guess = peak_width
                gl = 1
                params_init.extend([amp_guess, cen_guess, wid_guess, gl])
                bounds_low = [0, 0, 0, 0]
                bounds_up = [np.inf, np.inf, np.inf, 1]
                bounds_lower.extend(bounds_low)
                bounds_upper.extend(bounds_up)
            params_opt, pcov = curve_fit(lambda x, *params: multi_pseudoVoigt(x, *params), x_fit, y_fit, p0=params_init, bounds=(bounds_lower, bounds_upper))
            y_fit_curve = multi_pseudoVoigt(x_fit, *params_opt)
            y_subtracted = y_fit - y_fit_curve
            additional_peak_centers, additional_peak_intensities, additional_num_peaks = find_peaks_above_threshold(x_fit, y_subtracted, peak_width, 4 * threshold, peak_distance, 4 * peak_prominence)
            if additional_num_peaks > 0:
                peak_centers = np.append(peak_centers, additional_peak_centers)
                peak_intensities = np.append(peak_intensities, additional_peak_intensities)
                num_peaks = num_peaks + additional_num_peaks
                if num_peaks <= max_peaks:
                    n_peaks = num_peaks
                else:
                    n_peaks = max_peaks
                params_init = []
                bounds_lower = []
                bounds_upper = []
                for i in range(n_peaks):
                    amp_guess = peak_intensities[i]
                    cen_guess = peak_centers[i]
                    wid_guess = peak_width
                    gl = 0.5
                    params_init.extend([amp_guess, cen_guess, wid_guess, gl])
                    bounds_low = [0, 0, 0, 0]
                    bounds_up = [np.inf, np.inf, np.inf, 1]
                    bounds_lower.extend(bounds_low)
                    bounds_upper.extend(bounds_up)
                params_opt, pcov = curve_fit(lambda x, *params: multi_pseudoVoigt(x, *params), x_fit, y_fit, p0=params_init, bounds=(bounds_lower, bounds_upper))
                y_fit_curve = multi_pseudoVoigt(x_fit, *params_opt)
                print(len(y_fit_curve))
            else:
                pass
            peak_centers = [params_opt[4 * j + 1] for j in range(n_peaks)]
        return (x_fit, y_fit, background, background_fit, bkg_non_peaks_x, bkg_non_peaks_y, bkg_peaks_x, bkg_peaks_y, y_fit_curve, params_opt, pcov, peak_centers, y_subtracted)

def on_fit_button_click(event):
    global fit_results, peak_centers, max_peaks, current_index, group_tags
    try:
        max_peaks = int(max_peaks_text_box.text)
        peak_threshold = float(threshold_text_box.text)
        bkg_peak_threshold = float(bkg_peak_threshold_text_box.text)
        bkg_peak_prominence = float(bkg_peak_prominence_text_box.text)
        bkg_peak_width_multiplier = float(bkg_peak_width_multiplier_text_box.text)
        peak_width = float(peak_width_text_box.text)
        peak_prominence = float(peak_prominence_text_box.text)
    except ValueError:
        print('Invalid input for search settings.')
        return
    fit_results = {}
    peak_centers = {}
    current_index = 0
    if 'x_start' not in globals() or 'x_end' not in globals():
        print('No x-range selected.')
        return
    x_range = (min(x_start, x_end), max(x_start, x_end))
    tolerance = 0.05
    group_id = 0
    for i, file_path in enumerate(file_paths):
        data = np.loadtxt(file_path)
        x, y = (data[:, 0], data[:, 1])
        x_fit, y_fit, background, background_fit, bkg_non_peaks_x, bkg_non_peaks_y, bkg_peaks_x, bkg_peaks_y, y_fit_curve, params_opt, pcov, centers, y_subtracted = fit_peak_to_data(x, y, x_range, max_peaks, peak_threshold, bkg_peak_threshold, bkg_peak_prominence, bkg_peak_width_multiplier, peak_width, peak_prominence)
        base_name = os.path.basename(file_path)
        fit_result = {'Filename': base_name, 'peak_function': selected_function, 'x': x, 'y': y, 'x_fit': x_fit, 'y_fit': y_fit, 'background': background, 'background_fit': background_fit, 'bkg_non_peaks_x': bkg_non_peaks_x, 'bkg_non_peaks_y': bkg_non_peaks_y, 'bkg_peaks_x': bkg_peaks_x, 'bkg_peaks_y': bkg_peaks_y, 'y_fit_curve': y_fit_curve, 'y_subtracted': y_subtracted, 'params': params_opt, 'pcov': pcov, 'peak_centers': centers, 'groups': []}
        for center in np.sort(centers):
            fit_result['groups'].append(group_id)
            group_id += 1
        fit_results[i] = fit_result
        peak_centers[i] = centers
    for i in range(len(file_paths)):
        current_centers = peak_centers[i]
        for next_centers_id in range(i + 1, len(file_paths)):
            next_centers = peak_centers[next_centers_id]
            current_groups = fit_results[i]['groups']
            next_groups = fit_results[next_centers_id]['groups']
            for ci, current_center in enumerate(current_centers):
                for ni, next_center in enumerate(next_centers):
                    if abs(current_center - next_center) <= tolerance:
                        group_to_merge = next_groups[ni]
                        group_to_replace = current_groups[ci]
                        for idx, group in enumerate(next_groups):
                            if group == group_to_merge:
                                next_groups[idx] = group_to_replace
                        for j in range(next_centers_id + 1, len(file_paths)):
                            if group_to_merge in fit_results[j]['groups']:
                                fit_results[j]['groups'] = [group_to_replace if g == group_to_merge else g for g in fit_results[j]['groups']]
    unique_groups = sorted(set((g for result in fit_results.values() for g in result['groups'])))
    group_map = {old_group: new_group + 1 for new_group, old_group in enumerate(unique_groups)}
    for fit_result in fit_results.values():
        fit_result['groups'] = [group_map[g] for g in fit_result['groups']]
    display_combined_window()

def on_image_click(event):
    if event.inaxes == ax_intensity_map:
        y_index = int(round(event.ydata))
        if 0 <= y_index < len(fit_results):
            update_fit_plot(y_index)

def update_fit_plot(index):
    global current_index
    current_index = index
    ax_fit.clear()
    fit_data = fit_results[index]
    if background_enabled == False:
        ax_fit.plot(fit_data['x'], fit_data['y'], color='black', lw=1, label='Original Data')
        ax_fit.scatter(fit_data['bkg_non_peaks_x'], fit_data['bkg_non_peaks_y'], color='blue', marker='x', s=5, label='bkg data')
        ax_fit.scatter(fit_data['bkg_peaks_x'], fit_data['bkg_peaks_y'], color='green', marker='x', s=5, label='pseudo bkg data')
        ax_fit.plot(fit_data['x'], fit_data['background'], color='blue', lw=1, label='Background')
        try:
            ax_fit.scatter(fit_data['x_fit'], fit_data['y_fit'] + fit_data['background_fit'], color='red', marker='o', s=5, label='Data for Fit')
            ax_fit.plot(fit_data['x_fit'], fit_data['y_fit_curve'] + fit_data['background_fit'], color='red', ls='dashed', lw=1, label='Peak Fit')
            ax_fit.scatter(fit_data['x_fit'], fit_data['y_subtracted'] + fit_data['background_fit'], color='orange', marker='^', s=5, label='residual pre-refit')
        except:
            pass
    else:
        ax_fit.plot(fit_data['x'], fit_data['y'] - fit_data['background'], color='black', lw=1, label='Bkg subtracted Data')
        ax_fit.scatter(fit_data['x_fit'], fit_data['y_fit'], color='blue', marker='o', s=5, label='Data for Fit')
        try:
            ax_fit.plot(fit_data['x_fit'], fit_data['y_fit_curve'], color='red', ls='dashed', lw=1, label='Peak Fit')
        except:
            pass
    ax_fit.set_xlabel('2 theta (deg)')
    ax_fit.set_ylabel('Intensity (a.u.)')
    ax_fit.legend()
    ax_fit.set_title(f"Fit for {fit_data['Filename']}")
    ax_fit.set_ylim(-1 * np.max(fit_data['y']) * 0.05, np.max(fit_data['y']) * 1.2)
    fig_combined.canvas.draw()

def display_combined_window():
    global fig_combined, ax_intensity_map, ax_fit, current_index, background_button
    fig_combined, (ax_intensity_map, ax_fit) = plt.subplots(1, 2, figsize=(12, 6))
    fig_combined.suptitle('Intensity Map and Gaussian Fits')
    data_matrix = np.vstack([np.loadtxt(fp)[:, 1] for fp in file_paths])
    x_common = np.loadtxt(file_paths[0])[:, 0]
    vmin, vmax = np.percentile(data_matrix, [25, 99])
    cax = ax_intensity_map.imshow(data_matrix, aspect='auto', extent=[x_common[0], x_common[-1], 0, len(file_paths)], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax_intensity_map.set_xlabel('X Coordinate')
    ax_intensity_map.set_ylabel('Dataset Number')
    plt.colorbar(cax, ax=ax_intensity_map, label='Intensity')
    ax_intensity_map.set_title('Intensity Map with Peak Centers Overlay')
    fit_df = generate_peak_dataframe(fit_results, peak_centers)
    for column in fit_df.columns:
        if 'center' in column:
            if 'err' in column:
                pass
            else:
                x = fit_df[column]
                y = fit_df.index
                label = column.replace('_center', '')
                ax_intensity_map.scatter(x, y, s=5, label=label)
    fig_combined.canvas.mpl_connect('button_press_event', on_image_click)
    ax_prev = plt.axes([0.45, 0.02, 0.1, 0.05])
    ax_next = plt.axes([0.55, 0.02, 0.1, 0.05])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')

    def show_previous(event):
        if current_index > 0:
            update_fit_plot(current_index - 1)

    def show_next(event):
        if current_index < len(fit_results) - 1:
            update_fit_plot(current_index + 1)
    btn_prev.on_clicked(show_previous)
    btn_next.on_clicked(show_next)
    ax_button = plt.axes([0.85, 0.9, 0.1, 0.05])
    background_button = Button(ax_button, 'Bkg Off')
    background_button.on_clicked(toggle_background)
    ax_save = plt.axes([0.85, 0.02, 0.1, 0.05])
    save_button = Button(ax_save, 'Save Fit')
    save_button.on_clicked(lambda event: save_all_fit_results(event, fit_results, peak_centers))
    update_fit_plot(current_index)
    plt.show()

def generate_peak_dataframe(fit_results, peak_centers):
    rows = []
    for i, fit_result in fit_results.items():
        filename = fit_result['Filename']
        groups = fit_result['groups']
        params = fit_result['params']
        pcov = fit_result['pcov']
        peak_centers = fit_result['peak_centers']
        peak_func = fit_result['peak_function']
        num_peaks = len(peak_centers)
        if num_peaks > 0:
            perr = np.sqrt(np.diag(pcov))
        row = {'Filename': filename}
        for idx, group in enumerate(groups):
            group_prefix = f'Group{group}'
            if peak_func == 'Gaussian':
                amp, cen, wid = params[3 * idx:3 * idx + 3]
                amp_err, cen_err, wid_err = perr[3 * idx:3 * idx + 3]
            else:
                amp, cen, wid, gl = params[4 * idx:4 * idx + 4]
                amp_err, cen_err, wid_err, gl_err = perr[4 * idx:4 * idx + 4]
            row[f'{group_prefix}_amplitude'] = amp
            row[f'{group_prefix}_amplitude_err'] = amp_err
            row[f'{group_prefix}_center'] = cen
            row[f'{group_prefix}_center_err'] = cen_err
            if peak_func == 'Gaussian':
                row[f'{group_prefix}_width'] = wid
                row[f'{group_prefix}_width_err'] = wid_err
            else:
                row[f'{group_prefix}_width'] = wid
                row[f'{group_prefix}_width_err'] = wid_err
                row[f'{group_prefix}_gl'] = gl
                row[f'{group_prefix}_gl_err'] = gl_err
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def save_all_fit_results(event, fit_results, peak_centers):
    """
    # Combine all results into a single DataFrame
    all_data = pd.DataFrame()

    # Combine the dictionaries into a DataFrame
    for result in all_results:
        df = pd.DataFrame(result)
        all_data = pd.concat([all_data, df], ignore_index=True)
    """
    all_data = generate_peak_dataframe(fit_results, peak_centers)
    save_dir = 'fit_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'all_fit_results.xlsx')
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        all_data.to_excel(writer, index=False, sheet_name='Fit Results')
    print(f'All fit results saved to {save_path}')

def main():
    global file_paths, max_peaks_text_box, threshold_text_box, bkg_peak_threshold_text_box, bkg_peak_prominence_text_box, bkg_peak_width_multiplier_text_box, peak_width_text_box, peak_prominence_text_box, background_enabled, selected_function
    background_enabled = False
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title='Select data files', filetypes=[('XY files', '*.xy'), ('All files', '*.*')])
    root.destroy()
    if not file_paths:
        print('No files selected.')
        return
    data_matrices = []
    x_common = None
    for file_path in file_paths:
        data = np.loadtxt(file_path)
        x, y = (data[:, 0], data[:, 1])
        if x_common is None:
            x_common = x
        elif not np.allclose(x, x_common):
            raise ValueError('X coordinates in all datasets must match.')
        data_matrices.append(y)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    plt.subplots_adjust(bottom=0.25)
    data_matrix = np.vstack(data_matrices)
    vmin, vmax = np.percentile(data_matrix, [25, 99])
    cax = ax.imshow(data_matrix, aspect='auto', extent=[x_common[0], x_common[-1], 0, len(file_paths)], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Dataset Number')
    plt.colorbar(cax, label='Intensity')
    plt.title('Select x-range for Gaussian fit by dragging')
    rectangle_selector = RectangleSelector(ax, on_select_x_range, useblit=True, interactive=True)
    ax_radio = plt.axes([0.01, 0.02, 0.15, 0.1], facecolor='lightgrey')
    radio = RadioButtons(ax_radio, ('PseudoVoigt', 'Gaussian'))
    selected_function = 'PseudoVoigt'

    def update_peak_function(label):
        global selected_function
        selected_function = label
    radio.on_clicked(update_peak_function)
    bkg_peak_threshold_text_ax = plt.axes([0.29, 0.1, 0.05, 0.05])
    bkg_peak_threshold_text_box = TextBox(bkg_peak_threshold_text_ax, 'Bkg: threshold:', initial='2')
    bkg_peak_prominence_text_ax = plt.axes([0.46, 0.1, 0.05, 0.05])
    bkg_peak_prominence_text_box = TextBox(bkg_peak_prominence_text_ax, 'prominence:', initial='2')
    bkg_peak_width_multiplier_text_ax = plt.axes([0.66, 0.1, 0.05, 0.05])
    bkg_peak_width_multiplier_text_box = TextBox(bkg_peak_width_multiplier_text_ax, 'width multiplier:', initial='2')
    peak_width_text_ax = plt.axes([0.29, 0.02, 0.05, 0.05])
    peak_width_text_box = TextBox(peak_width_text_ax, 'Peaks: width:', initial='0.25')
    threshold_text_ax = plt.axes([0.46, 0.02, 0.05, 0.05])
    threshold_text_box = TextBox(threshold_text_ax, 'threshold:', initial='5')
    peak_prominence_text_ax = plt.axes([0.66, 0.02, 0.05, 0.05])
    peak_prominence_text_box = TextBox(peak_prominence_text_ax, 'prominence:', initial='5')
    max_peaks_text_ax = plt.axes([0.8, 0.02, 0.05, 0.05])
    max_peaks_text_box = TextBox(max_peaks_text_ax, 'Max pks:', initial='1')
    ax_fit_button = plt.axes([0.92, 0.02, 0.05, 0.05])
    fit_button = Button(ax_fit_button, 'Fit')
    fit_button.on_clicked(on_fit_button_click)
    plt.show()

