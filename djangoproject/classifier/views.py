# classifier/views.py

from django.shortcuts import render
from .forms import NewsForm
from .predict import predict_relationship  # 这应该是你的预测函数

def index(request):
    if request.method == 'POST':
        form = NewsForm(request.POST)
        if form.is_valid():
            headline = form.cleaned_data['headline']
            body = form.cleaned_data['body']
            prediction = predict_relationship(headline, body)
            return render(request, 'result.html', {'prediction': prediction})
    else:
        form = NewsForm()
    return render(request, 'index.html', {'form': form})