`timescale 1ns / 1ps

module tb_alarm_system;

    reg [7:0] temp_sensor;
    reg smoke_sensor;
    wire alarm_out;

    // Instantiate the Unit Under Test (UUT)
    alarm_system uut (
        .temp_sensor(temp_sensor), 
        .smoke_sensor(smoke_sensor), 
        .alarm_out(alarm_out)
    );

    // Clock and reset generation
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    reg reset;
    initial begin
        reset = 1;
        #15 reset = 0;
    end

    // Test cases
    initial begin
        // Monitor changes on important signals
        $monitor("Time = %t, temp_sensor = %d, smoke_sensor = %b, alarm_out = %b",
                  $time, temp_sensor, smoke_sensor, alarm_out);

        // Initialize Inputs
        temp_sensor = 0;
        smoke_sensor = 0;

        // Wait for reset deassertion
        @(negedge reset);
        #10;

        // Test Case 1: No smoke, temperature below threshold (should not trigger alarm)
        temp_sensor = 8'd50; // Below the typical threshold
        smoke_sensor = 1'b0;
        #20;
        check_alarm(1'b0);

        // Test Case 2: No smoke, temperature above threshold (should trigger alarm)
        temp_sensor = 8'd100; // Above the typical threshold
        smoke_sensor = 1'b0;
        #20;
        check_alarm(1'b1);

        // Test Case 3: Smoke detected, irrespective of temperature (should trigger alarm)
        temp_sensor = 8'd25; // Irrelevant
        smoke_sensor = 1'b1;
        #20;
        check_alarm(1'b1);

        // Test Case 4: Smoke detected and temperature above threshold (should trigger alarm)
        temp_sensor = 8'd100; // Above the typical threshold
        smoke_sensor = 1'b1;
        #20;
        check_alarm(1'b1);

        // Test Case 5: Random high-stress temperatures and smoke combinations
        repeat (10) begin
            temp_sensor = $random;
            smoke_sensor = $random % 2;
            #20;
            check_alarm((temp_sensor > 90 || smoke_sensor == 1'b1) ? 1'b1 : 1'b0);
        end

        // Conclude tests
        $display("===========Your Design Passed===========");
        $finish;
    end

    task check_alarm;
        input expected_alarm;
        begin
            if (alarm_out !== expected_alarm) begin
                $display("===========Error at time %t===========", $time);
                $display("Temp: %d, Smoke: %b, Expected Alarm: %b, Actual Alarm: %b", 
                        temp_sensor, smoke_sensor, expected_alarm, alarm_out);
                $finish;
            end
        end
    endtask

endmodule
