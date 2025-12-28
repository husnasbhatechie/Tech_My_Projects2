from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained energy consumption model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

EMISSION_FACTOR = 0.82  # kg CO2 per kWh

@app.route("/", methods=["GET", "POST"])
def index():
    single_result = None
    batch_table = None
    energy_chart = None
    co2_chart = None

    # ---------- SINGLE HOUSEHOLD ----------
    if request.method == "POST" and 'single_submit' in request.form:
        try:
            household_size = float(request.form['Household_Size'])
            monthly_income = float(request.form['Monthly_Income'])
            appliance_count = int(request.form['Appliance_Count'])
            avg_daily_hours = float(request.form['Avg_Daily_Usage_Hours'])

            X = [[household_size, monthly_income, appliance_count, avg_daily_hours]]
            predicted = round(model.predict(X)[0],2)

            # Consumption Level & Recommendation
            if predicted < 100:
                level = "Low"
                energy_recommendation = "Efficient usage. Maintain current habits."
            elif predicted <= 200:
                level = "Medium"
                energy_recommendation = "Reduce daily usage hours and switch off idle appliances."
            else:
                level = "High"
                energy_recommendation = "Use energy-efficient appliances and consider solar energy."

            # Estimate CO2
            co2_emission = round(predicted * EMISSION_FACTOR, 2)
            if co2_emission < 100:
                co2_level_val = "Low"
                co2_recommendation = "Good job! Maintain your energy habits."
            elif co2_emission <= 300:
                co2_level_val = "Medium"
                co2_recommendation = "Consider reducing usage of high-power appliances."
            else:
                co2_level_val = "High"
                co2_recommendation = "Switch to energy-efficient appliances or solar energy."

            single_result = {
                "Predicted_Consumption": predicted,
                "Consumption_Level": level,
                "Energy_Recommendation": energy_recommendation,
                "CO2_Emission_kg": co2_emission,
                "CO2_Level": co2_level_val,
                "CO2_Recommendation": co2_recommendation
            }
        except Exception as e:
            single_result = {"error": str(e)}

    # ---------- BATCH PREDICTION ----------
    if request.method == "POST" and 'batch_submit' in request.form:
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
            
            # Check CSV type
            if all(col in df.columns for col in ["Household_ID","Appliance_Type","Power_kW","Avg_Daily_Usage_Hours","Days_Per_Month"]):
                # Appliance-level CSV
                df["CO2_kg"] = df["Power_kW"] * df["Avg_Daily_Usage_Hours"] * df["Days_Per_Month"] * EMISSION_FACTOR
                co2_per_house = df.groupby("Household_ID")["CO2_kg"].sum().reset_index()
                co2_per_house.rename(columns={"CO2_kg":"CO2_Emission_kg"}, inplace=True)

                # Energy prediction if household features exist
                features = ["Household_Size","Monthly_Income","Appliance_Count","Avg_Daily_Usage_Hours"]
                if all(f in df.columns for f in features):
                    df_features = df.groupby("Household_ID").agg({
                        "Household_Size":"first",
                        "Monthly_Income":"first",
                        "Appliance_Count":"first",
                        "Avg_Daily_Usage_Hours":"mean"
                    }).reset_index()
                    df_features["Predicted_Consumption"] = model.predict(df_features[features]).round(2)
                else:
                    df_features = co2_per_house.copy()
                    df_features["Predicted_Consumption"] = None

                df_final = pd.merge(df_features, co2_per_house, on="Household_ID", how="left")

            elif all(col in df.columns for col in ["Household_ID","Household_Size","Monthly_Income","Appliance_Count","Avg_Daily_Usage_Hours"]):
                # Household-level CSV
                X = df[["Household_Size","Monthly_Income","Appliance_Count","Avg_Daily_Usage_Hours"]].astype(float)
                df["Predicted_Consumption"] = model.predict(X).round(2)
                df["CO2_Emission_kg"] = (df["Predicted_Consumption"] * EMISSION_FACTOR).round(2)
                df_final = df.copy()

            else:
                return "CSV format not recognized. Upload appliance-level or household-level CSV."

            # Consumption Level & Recommendation
            def energy_level(val):
                if pd.isnull(val):
                    return "-"
                elif val < 100:
                    return "Low"
                elif val <= 200:
                    return "Medium"
                else:
                    return "High"
            df_final["Consumption_Level"] = df_final["Predicted_Consumption"].apply(energy_level)

            def energy_recommendation(level):
                if level=="Low":
                    return "Efficient usage. Maintain current habits."
                elif level=="Medium":
                    return "Reduce daily usage hours and switch off idle appliances."
                elif level=="High":
                    return "Use energy-efficient appliances and consider solar energy."
                else:
                    return "-"
            df_final["Energy_Recommendation"] = df_final["Consumption_Level"].apply(energy_recommendation)

            # CO2 Level & Recommendation
            def co2_level(val):
                if pd.isnull(val):
                    return "-"
                elif val < 100:
                    return "Low"
                elif val <= 300:
                    return "Medium"
                else:
                    return "High"
            df_final["CO2_Level"] = df_final["CO2_Emission_kg"].apply(co2_level)

            def co2_recommendation(level):
                if level=="Low":
                    return "Good job! Maintain your energy habits."
                elif level=="Medium":
                    return "Consider reducing usage of high-power appliances."
                elif level=="High":
                    return "Switch to energy-efficient appliances or solar energy."
                else:
                    return "-"
            df_final["CO2_Recommendation"] = df_final["CO2_Level"].apply(co2_recommendation)

            # Save CSV
            os.makedirs("static", exist_ok=True)
            df_final.to_csv("static/batch_prediction_output.csv", index=False)

            # Charts
            plt.figure(figsize=(8,4))
            plt.bar(df_final["Household_ID"], df_final["Predicted_Consumption"], color="steelblue")
            plt.xlabel("Household ID")
            plt.ylabel("Predicted Energy (kWh)")
            plt.title("Predicted Energy Consumption")
            plt.tight_layout()
            plt.savefig("static/energy_bar_chart.png")
            plt.close()

            plt.figure(figsize=(8,4))
            plt.bar(df_final["Household_ID"], df_final["CO2_Emission_kg"], color="green")
            plt.xlabel("Household ID")
            plt.ylabel("CO₂ Emission (kg)")
            plt.title("CO₂ Emission per Household")
            plt.tight_layout()
            plt.savefig("static/co2_bar_chart.png")
            plt.close()

            batch_table = df_final.to_html(index=False, classes="table table-striped table-bordered table-sm")
            energy_chart = "energy_bar_chart.png"
            co2_chart = "co2_bar_chart.png"

    return render_template("index.html",
                           single_result=single_result,
                           batch_table=batch_table,
                           energy_chart=energy_chart,
                           co2_chart=co2_chart)

# ----- DOWNLOAD BATCH CSV -----
@app.route("/download_batch")
def download_batch():
    path = "static/batch_prediction_output.csv"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "No batch prediction file found."

if __name__ == "__main__":
    app.run(debug=True)
