from django.shortcuts import render

def base_file(request):
    return render(request, "verifier/base.html")  # Looks inside templates/verifier/templates/
