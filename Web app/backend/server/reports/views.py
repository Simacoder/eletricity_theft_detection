from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Report

class GenerateReport(APIView):
    def post(self, request):
        report = Report.objects.create(
            report_type=request.data.get('report_type'),
            content="Generated report content"
        )
        return Response({"message": f"Report {report.report_type} generated."})
