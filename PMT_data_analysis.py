import pandas as pd
import numpy as np
from scipy.signal import find_peaks

Peak_diff = 4.0

def PMT_data_analysis(pmt_folder_path, output_csv_path):
    # Load the CSV file
    df = pd.read_csv(pmt_folder_path, skiprows=10)  # Skip the first row if it contains metadata

    # Ensure CH1 exists
    # if "CH1" not in df.columns or "Source" not in df.columns:
    #     raise ValueError("CSV must contain 'Source' and 'CH1' columns")

    time = df["Source"].values[1:-1].astype(float)
    ch1 = df["CH1"].values[1:-1].astype(float)
    ch2 = df["CH2"].values[1:-1].astype(float)

    # Invert signal to detect negative dips as peaks
    inverted_signal = -ch1

    # Find start time where CH2 near 0.0
    for i in range(len(ch2)):
        if ch2[i] < 1.0:
            start_time = time[i]
            break
        
    time = time - start_time

    # Detect dips (negative peaks)
    peaks, properties = find_peaks(inverted_signal, prominence=Peak_diff)  # adjust prominence if needed

    results = []
    n = 0 # Number of dips
    index_prev = -100 # Previous peak index
    
    for i, peak in enumerate(peaks):
        ## Condition to skip peak will be changed based on requirements (TBD)
        if peak != 0 and peak - index_prev < 50:
            continue # Skip if too close to previous peak

        peak_time = time[peak]
        peak_voltage = ch1[peak]

        # Delta is the difference between peak voltage and mean voltage used to determine duration
        Delta = abs(peak_voltage - np.mean(ch1))
        duration = 0.0
        

        # Calculate duration
        for j in range(peak, len(ch1) - 1, 1):
            if abs(ch1[peak] - ch1[j]) > (Delta - 1.0): # Condition to find end of dip duration
                # print(peak, j)
                duration = time[j] - time[peak]
                break

        n += 1
        results.append({
            "Number": n,
            "Index": peak,
            "Time (s)": peak_time,
            "Voltage (V)": peak_voltage,
            "Duration (s)": duration
        })

        index_prev = peak
        # Append an empty dictionary for the next peak
        results.append({})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    print(f"Saved negative dip analysis to {output_csv_path}")
    return results_df

# Example usage:
if __name__ == "__main__":
    PMT_data_analysis("c:/Users/arnob/OneDrive/Desktop/Zucrow Lab Research/Arnob//TEST5.csv", 
                      "c:/Users/arnob/OneDrive/Desktop/Zucrow Lab Research/Arnob//TEST5_Output.csv")
