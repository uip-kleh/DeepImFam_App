from django.http import HttpResponse
from django.template import loader
from .forms import AminoForm, PhotoForm
from .models import *
from django.shortcuts import render, redirect
import numpy as np
import matplotlib.pylab as plt  
from io import BytesIO

# Create your views here.
def index(request):
    template = loader.get_template("imageai/index.html")
    context = {'form': AminoForm()}
    return HttpResponse(template.render(context, request))

def about(request):
    template = loader.get_template('imageai/about.html')
    context = dict()
    return HttpResponse(template.render(context, request))

def contact(request):
    template = loader.get_template('imageai/contact.html')
    context = dict()
    return HttpResponse(template.render(context, request))

def predict(request):
    if not request.method == 'GET': return redirect('imageai:index')
    form = AminoForm(request.GET)
    print(form)
    if not form.is_valid(): raise ValueError('Form is not valid!')

    aaseq = form.cleaned_data['aa_seq']
    deepimfam = DeepImFam(request.POST, aaseq=aaseq)
    #aaimg_data = deepimfam.generate_img()
    pred_label, percentage, pred_sub_label, percentage_sub, pred_subsub_label, percentage_subsub, graph = deepimfam.predict()
    #generate_img(aaseq)

    template = loader.get_template('imageai/result.html')
    context = {
        'aaseq': aaseq,
        'graph': graph,
        'pred_label': pred_label,
        'percentage': percentage,
        'pred_sub_label': pred_sub_label,
        'percentage_sub': percentage_sub,
        'pred_subsub_label': pred_subsub_label,
        'percentage_subsub': percentage_subsub,
    }

    return HttpResponse(template.render(context, request))