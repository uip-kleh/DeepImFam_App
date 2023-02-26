from django import forms

class AminoForm(forms.Form):
    aa_seq = forms.CharField(
        max_length=1000,
    )

class PhotoForm(forms.Form):
    image=forms.ImageField(widget=forms.FileInput(attrs={"class":"custom-file-input"}))