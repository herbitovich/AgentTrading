from django.db import models

class Trade(models.Model):
    date = models.DateTimeField()
    TRADE_ACTIONS = (
        ("buy", "Buy"),
        ("sell", "Sell"),
        ("hold", "Hold"),
    )
    action = models.CharField(
        max_length=4,
        choices=TRADE_ACTIONS,
    )
    amount = models.FloatField(null=True, blank=True)
    current_price = models.FloatField()
    company = models.CharField(max_length=256)
    agent = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.company} - {self.agent} - {self.action} - {self.amount} - {self.date}"