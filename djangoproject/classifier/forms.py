# classifier/forms.py

from django import forms

class NewsForm(forms.Form):
    headline = forms.CharField(label='Headline', max_length=1000)
    body = forms.CharField(label='Body', max_length=10000, widget=forms.Textarea)