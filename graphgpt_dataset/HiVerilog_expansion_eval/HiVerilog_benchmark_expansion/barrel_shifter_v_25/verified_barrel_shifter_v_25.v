module alarm_system (temp_sensor, smoke_sensor, alarm_out);
    input  [7:0] temp_sensor;
    input  smoke_sensor;
    output alarm_out;
    wire temp_high;

    // Instance of temperature comparison
    compare_temp  comp_temp (.temp_in(temp_sensor), .temp_high(temp_high));

    // Instance of OR logic to determine alarm state
    or_logic  or_gate (.temp_high(temp_high), .smoke_signal(smoke_sensor), .alarm_out(alarm_out));
endmodule

module compare_temp(temp_in, temp_high);
    input [7:0] temp_in;
    output temp_high;
    // Temperature threshold for comparison
    parameter THRESHOLD = 8'h5A; // example threshold value (90 in decimal)
    
    assign temp_high = (temp_in > THRESHOLD) ? 1'b1 : 1'b0;
endmodule

module or_logic(temp_high, smoke_signal, alarm_out);
    input temp_high, smoke_signal;
    output alarm_out;
    
    assign alarm_out = temp_high | smoke_signal;
endmodule
