# app.py

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

# Import the N-dimensional wavelet transform functions
import wavelet_transform as wt

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated_coeffs'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.secret_key = 'super_secret_key_for_nd'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # --- Form Validation ---
        if 'file' not in request.files:
            flash('No file part in request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if not file or not allowed_file(file.filename):
            flash('Invalid file type. Please upload a .csv file.')
            return redirect(request.url)

        try:
            levels = int(request.form['levels'])
            data_dimension = request.form['data_dimension']
            if levels < 1:
                flash('Decomposition levels must be at least 1.')
                return redirect(request.url)
        except (ValueError, TypeError, KeyError):
            flash('Invalid form data.')
            return redirect(request.url)

        # --- File and Data Processing ---
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            
            # ==================================================================
            # FIX: Impute missing values by filling with the column's mean
            # ==================================================================
            df.fillna(df.mean(), inplace=True)


            data = None

            # Load data based on user-selected dimension
            if data_dimension == '1d':
                column_name = request.form.get('column_name')
                if not column_name:
                    flash('Column Name is required for 1D data.')
                    return redirect(request.url)
                if column_name not in df.columns:
                    flash(f"Column '{column_name}' not found in the file.")
                    return redirect(request.url)
                data = df[column_name].values

            elif data_dimension == '2d':
                # For 2D, we use all numerical columns from the CSV
                numeric_df = df.select_dtypes(include=np.number)
                if numeric_df.empty:
                    flash('No numeric data found in the CSV for 2D processing.')
                    return redirect(request.url)
                data = numeric_df.to_numpy()
                flash(f'Processing 2D data with shape: {data.shape}', 'info')
            
            else:
                flash('Invalid data dimension specified.')
                return redirect(request.url)

            # --- N-Dimensional Wavelet Transform ---
            coeffs_tree = wt.haar_lwt_nd_decompose(data, level=levels)

            # --- Save Coefficients and Get DataFrame for Display ---
            output_filename = f"coeffs_{data_dimension}D_{os.path.splitext(filename)[0]}.csv"
            coeffs_df = wt.get_coefficients_df_and_save(
                app.config['GENERATED_FOLDER'],
                coeffs_tree,
                output_filename,
                levels
            )

            coeffs_html = coeffs_df.to_html(classes='table table-striped table-hover', index=False, na_rep='-')

            # --- Reconstruction Check (for verification) ---
            reconstructed_data = wt.haar_lwt_nd_reconstruct(coeffs_tree)
            is_accurate = np.allclose(data, reconstructed_data)

            return render_template('results.html',
                                   table=coeffs_html,
                                   filename=output_filename,
                                   is_accurate=is_accurate)

        except Exception as e:
            flash(f'An error occurred: {e}')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(GENERATED_FOLDER, exist_ok=True)
    app.run(debug=True)