from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Trade
from .serializers import TradeSerializer

@api_view(['POST'])
def post_trades(request):
    serializer = TradeSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_trades(request, company):
    n = int(request.GET.get('n', 30))
    trades = Trade.objects.filter(company__iexact=company).order_by('-date')[:n]

    if not trades:
        return Response({"message": f"No trades were found for the company {company}"}, status=status.HTTP_404_NOT_FOUND)
    serializer = TradeSerializer(trades, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)



