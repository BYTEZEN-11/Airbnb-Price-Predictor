import io
import math
import os
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from Airbnb.logger import logging
from Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)
app.secret_key = os.environ.get("AIRBNB_SECRET_KEY", "super-secret-airbnb-key")


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=()"
    return response


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    try:
        data = request.form

        propertytype = data.get("propertytype") or data.get("property_type")
        roomtype = data.get("roomtype") or data.get("room_type")
        city = data.get("city")
        accommodates = data.get("accommodates")
        bedrooms = data.get("bedrooms")
        beds = data.get("beds")
        bathrooms = data.get("bathrooms")
        amenties = data.get("amenties") if "amenties" in data else data.get("amenities")
        bedtype = data.get("bedtype") or data.get("bed_type")
        canceltype = data.get("canceltype") or data.get("cancellation_policy")
        clean = data.get("clean") or data.get("cleaning_fee")
        instbook = data.get("instbook") or data.get("instant_bookable")
        dp = data.get("dp") or data.get("host_has_profile_pic")
        verify = data.get("verify") or data.get("host_identity_verified")
        hostresponse = data.get("hostresponse") or data.get("host_response_rate")
        lat = data.get("lat") or data.get("latitude")
        long_val = data.get("long") or data.get("longitude")
        review = data.get("review") or data.get("number_of_reviews")
        overallreview = data.get("overallreview") or data.get("review_scores_rating")

        required_fields = [
            propertytype,
            roomtype,
            city,
            accommodates,
            bedrooms,
            beds,
            bathrooms,
            amenties,
            bedtype,
            canceltype,
            clean,
            instbook,
            dp,
            verify,
            hostresponse,
            lat,
            long_val,
            review,
            overallreview,
        ]

        if any(f is None or str(f).strip() == "" for f in required_fields):
            return render_template("error.html", error_message="Missing required form fields"), 400

        custom_data = CustomData(
            property_type=str(propertytype),
            room_type=str(roomtype),
            city=str(city),
            accommodates=int(accommodates),
            bedrooms=int(bedrooms),
            beds=int(beds),
            bathrooms=float(bathrooms),
            amenities=int(amenties),
            bed_type=str(bedtype),
            cancellation_policy=str(canceltype),
            cleaning_fee=str(clean),
            instant_bookable=str(instbook),
            host_has_profile_pic=str(dp),
            host_identity_verified=str(verify),
            host_response_rate=float(str(hostresponse).replace("%", "")),
            latitude=float(lat),
            longitude=float(long_val),
            number_of_reviews=int(review),
            review_scores_rating=float(overallreview),
        )

        pred_df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline(city=str(city))
        results = predict_pipeline.predict(pred_df)

        point_log = float(results[0])
        lo_log = float(results[1])
        hi_log = float(results[2])

        point_usd = math.expm1(point_log)
        lo_usd = math.expm1(lo_log)
        hi_usd = math.expm1(hi_log)

        final_result = f"{point_usd:.2f}"
        low_price = f"{lo_usd:.2f}"
        high_price = f"{hi_usd:.2f}"

        return render_template(
            "index.html",
            final_result=final_result,
            low_price=low_price,
            high_price=high_price,
        )

    except (ValueError, TypeError) as e:
        logging.error(f"Form parsing error: {e}")
        return render_template("error.html", error_message=f"Invalid form input data: {e}"), 400
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template("error.html", error_message=f"An error occurred during prediction: {e}"), 400


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {e}"}), 400

    req_cols = [
        "property_type",
        "room_type",
        "amenities",
        "accommodates",
        "bathrooms",
        "bed_type",
        "cancellation_policy",
        "cleaning_fee",
        "city",
        "host_has_profile_pic",
        "host_identity_verified",
        "host_response_rate",
        "latitude",
        "longitude",
        "number_of_reviews",
        "review_scores_rating",
        "bedrooms",
        "beds",
    ]

    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {', '.join(missing)}"}), 400

    results = []
    for idx, row in df.iterrows():
        try:
            city_val = str(row["city"])
            pipe = PredictPipeline(city=city_val)

            amen_val = row["amenities"]
            if isinstance(amen_val, str) and (amen_val.startswith("{") or "," in amen_val):
                amen_count = len(amen_val.strip("{}").split(","))
            else:
                amen_count = int(amen_val)

            hr_val = row["host_response_rate"]
            if isinstance(hr_val, str):
                hr_val = float(hr_val.replace("%", ""))
            else:
                hr_val = float(hr_val)

            cd = CustomData(
                property_type=str(row["property_type"]),
                room_type=str(row["room_type"]),
                amenities=amen_count,
                accommodates=int(row["accommodates"]),
                bathrooms=float(row["bathrooms"]),
                bed_type=str(row["bed_type"]),
                cancellation_policy=str(row["cancellation_policy"]),
                cleaning_fee=str(row["cleaning_fee"]),
                city=city_val,
                host_has_profile_pic=str(row["host_has_profile_pic"]),
                host_identity_verified=str(row["host_identity_verified"]),
                host_response_rate=hr_val,
                instant_bookable=str(row.get("instant_bookable", "f")),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                number_of_reviews=int(row["number_of_reviews"]),
                review_scores_rating=float(row["review_scores_rating"]),
                bedrooms=int(row["bedrooms"]),
                beds=int(row["beds"]),
            )
            row_df = cd.get_data_as_dataframe()
            pred = pipe.predict(row_df)
            pred_usd = math.expm1(float(pred[0]))
            results.append(round(pred_usd, 2))
        except Exception as e:
            logging.error(f"Row {idx} prediction error: {e}")
            results.append(None)

    df["predicted_price_usd"] = results
    output_buf = io.StringIO()
    df.to_csv(output_buf, index=False)
    return Response(
        output_buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "status": "healthy",
        "service": "Airbnb Price Predictor",
        "version": "1.0.0",
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
