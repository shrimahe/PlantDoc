from flask import Flask, render_template, request, redirect, flash

app = Flask(__name__)
app.secret_key = "plantixsecretkey"  # Required for flash messages

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/features')
def features():
    return render_template("features.html")

@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        # You could save this to a database or send an email here
        print(f"Received message from {name} ({email}): {message}")

        flash("Thank you! Your message has been received.")
        return redirect("/contact")

    return render_template("contact.html")
if __name__ == "__main__":
    app.run(debug=True)
