# FoodHub Order Status and ETA Guide

This guide explains what each order status and each time field on a FoodHub order means, so that questions such as "where is my order?" can be answered correctly from the order record.

## Overview
An order record contains the order ID, the customer ID, the time the order was placed, the order status, the payment status, the items, and several time fields: the preparation ETA, the prepared time, the delivery ETA and the delivery time. A field that is empty means that the event has not happened yet, or that it will never happen for that order.

## Order statuses

### preparing food
The restaurant has accepted the order and is cooking it. The order has a preparation ETA, which is when the restaurant expects the food to be ready. The prepared time, the delivery ETA and the delivery time are normally empty because those events have not happened yet.

### picked up
The food is ready and the delivery partner has collected it. The order has a prepared time and a delivery ETA. The delivery time is empty until the order arrives. The customer should keep their phone reachable.

### delivered
The order has reached the customer. All the time fields are filled, including the delivery time, which is when the order was handed over. A delivered order cannot be cancelled and any problem is handled through the Refund Policy.

### canceled
The order was cancelled before delivery. The preparation, prepared, delivery ETA and delivery times are normally empty because the order was stopped. Cancelled orders are not charged when cancelled in time.

## Time fields

### Order time
The time the customer placed the order.

### Preparation ETA and prepared time
The preparation ETA is when the restaurant expects the food to be ready. The prepared time is when the food was actually ready. If the prepared time is empty, the food is not ready yet.

### Delivery ETA and delivery time
The delivery ETA is when the order is expected to arrive after pickup. The delivery time is when the order was actually delivered. If the delivery time is empty, the order has not been delivered yet.

## Reading an order

### Order is still being prepared
When the status is "preparing food", tell the customer the order is being prepared and give the preparation ETA from the record. Do not invent a delivery time, because there is no delivery ETA yet.

### Order is on the way
When the status is "picked up", tell the customer the order is on its way and give the delivery ETA from the record. The order has not arrived, so there is no delivery time yet.

### Order has arrived
When the status is "delivered", give the delivery time from the record. If the customer says they did not receive it, a support agent checks the delivery record.

### Order was cancelled
When the status is "canceled", tell the customer the order was cancelled and that nothing is charged for a cancelled order. Do not give delivery information.

## Typical timings
Preparation usually takes 10 to 30 minutes and delivery usually takes 15 to 40 minutes after pickup, depending on the restaurant and distance. These are typical values only. The ETAs in the order record always take priority over typical timings.

## Frequently asked questions
- Why is my delivery time empty? The order has not been delivered yet, or it was cancelled.
- Why is there no delivery ETA? The food is probably still being prepared.
- What does the prepared time mean? It is when the restaurant finished cooking; the delivery partner collects the food after that.
