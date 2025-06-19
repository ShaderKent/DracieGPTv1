from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from main import query_alpaca

# Create your views here.
@api_view(['GET'])
def get_query(request):
    query = request.query_params.get("query", None)
    answer = query_alpaca(query, "", 10, 1024, 1.4, 10)
    JSON_response = {"value": answer}
    return Response(JSON_response)
