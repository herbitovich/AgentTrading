# Generated by Django 5.2 on 2025-04-13 18:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Trade',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('action', models.CharField(choices=[('buy', 'Buy'), ('sell', 'Sell'), ('hold', 'Hold')], max_length=4)),
                ('amount', models.FloatField(blank=True, null=True)),
                ('current_price', models.FloatField()),
                ('company', models.CharField(max_length=256)),
            ],
        ),
    ]
