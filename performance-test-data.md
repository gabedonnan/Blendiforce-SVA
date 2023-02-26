# Performance Testing

## Commit 8270cf3: 26/02/2023
**force_object_from_raw:** on ***n*** objects with number of vertices ***v*** over ***i*** iterations
> n = 360, v = 8, i = 1000
>> 0.027365 seconds per iteration
>> 
> n = 42, v = 162, i = 100
>> 0.073795 seconds per iteration
>> 
> n = 40, v = 507, i = 100
>> 0.185084 seconds per iteration

**force_object_from_raw_async:** on ***n*** objects with number of vertices ***v*** over ***i*** iterations
> n = 360, v = 8, i = 1000
>> 0.034428 seconds per iteration
>> 
> n = 42, v = 162, i = 100
>> 0.075735 seconds per iteration
>> 
> n = 40, v = 507, i = 100
>> 0.193236 seconds per iteration

### Notes:
Asynchronous operations unexpectedly slower for all instances, though as v increases the performance draws more in line with non asyncrhonous version

This is most likely due to the overhead calling and gathering each of the results of these functions.

For more performance bottlenecking functions the asyncrhonous versions will probably provide a speed-up instead of a slow-down
