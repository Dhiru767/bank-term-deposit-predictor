from django import forms

from ml import config as cfg

MONTH_NAMES = {
    "mar": "March", "apr": "April", "may": "May", "jun": "June",
    "jul": "July", "aug": "August", "sep": "September",
    "oct": "October", "nov": "November", "dec": "December",
}

LABEL_OVERRIDES = {
    "month": MONTH_NAMES,
    "contact": {"cellular": "Cellular (mobile)", "telephone": "Telephone (landline)"},
    "poutcome": {
        "failure": "Failure",
        "nonexistent": "Not contacted before",
        "success": "Success",
    },
}


def pretty(value):
    """'basic.4y' -> 'Basic 4Y', 'blue-collar' -> 'Blue Collar'"""
    return value.replace(".", " ").replace("-", " ").strip().title()


def choices_for(name):
    """Dropdown options built from ml/config.py, so form and model always agree."""
    labels = LABEL_OVERRIDES.get(name, {})
    options = [(value, labels.get(value, pretty(value))) for value in cfg.CHOICES[name]]
    return [("", "Select...")] + options


class PredictionForm(forms.Form):
    age = forms.IntegerField(
        min_value=18, max_value=100, label="Age",
        help_text="Client age in years (18 to 100).",
    )
    job = forms.ChoiceField(choices=choices_for("job"), label="Job")
    marital = forms.ChoiceField(choices=choices_for("marital"), label="Marital status")
    education = forms.ChoiceField(choices=choices_for("education"), label="Education")
    housing = forms.ChoiceField(choices=choices_for("housing"), label="Has a housing loan?")
    loan = forms.ChoiceField(choices=choices_for("loan"), label="Has a personal loan?")
    contact = forms.ChoiceField(choices=choices_for("contact"), label="Contact type")
    month = forms.ChoiceField(choices=choices_for("month"), label="Month of last contact", help_text="Our data covers March to December.")
    campaign = forms.IntegerField(
        min_value=1, max_value=50, label="Contacts in this campaign",
        help_text="Times the client was contacted in this campaign, including this call.",
    )
    previous = forms.IntegerField(
        min_value=0, max_value=20, label="Contacts before this campaign",
        help_text="Times the client was contacted in earlier campaigns (0 if never).",
    )
    poutcome = forms.ChoiceField(
        choices=choices_for("poutcome"), label="Outcome of the previous campaign",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            css = "form-select" if isinstance(field, forms.ChoiceField) else "form-control"
            field.widget.attrs["class"] = css
