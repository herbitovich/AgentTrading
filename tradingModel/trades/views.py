from collections import defaultdict
from rest_framework import status
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from .models import Trade
from .serializers import TradeSerializer
from .utils import company_stats

@api_view(['POST'])
@csrf_exempt
def post_trades(request):
    data = request.data
    company = data['company']
    current_price = data['current_price']
    date = data['date']
    agents = data['agents']

    trades = []
    for agent_data in agents:
        trade_data = {
            'company': company,
            'current_price': current_price,
            'date': date,
            'agent': agent_data.get('agent'),
            'action': agent_data.get('action'),
            'amount': agent_data.get('amount'),
            'value': agent_data.get('value')
        }
        serializer = TradeSerializer(data=trade_data)
        if serializer.is_valid():
            trades.append(serializer)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    for serializer in trades:
        serializer.save()
    return Response([s.data for s in trades], status=status.HTTP_201_CREATED)

@api_view(['GET'])
def get_trades(request, company):
    n = int(request.GET.get('n', 60))
    trades = Trade.objects.filter(company__iexact=company).order_by('-date')[:n]
    if not trades:
        return Response({"message": f"No trades were found for the company {company}"}, status=status.HTTP_404_NOT_FOUND)
    trades_list = list(reversed(trades))
    group = defaultdict(lambda: {"company" : company, "agents" : []})
    for trade in trades_list:
        key = trade.date.isoformat()
        if "date" not in group[key]:
            group[key]["date"] = trade.date.isoformat()
            group[key]["current_price"] = trade.current_price
        group[key]["agents"].append({
            "agent": trade.agent,
            "action": trade.action,
            "amount": trade.amount,
            "value": trade.value
        })
    stats = company_stats(trades_list)
    response_data = list(group.values())
    return Response({
        "trades": response_data,
        "stats": stats
    }, status=status.HTTP_200_OK)



