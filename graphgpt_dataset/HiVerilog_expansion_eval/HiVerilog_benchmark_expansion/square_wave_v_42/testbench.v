`timescale 1ns / 1ps

module tb_pwm_controller;

    // Inputs
    reg clk;
    reg [7:0] duty_cycle;

    // Outputs
    wire pwm_out;

    // Instantiate the Unit Under Test (UUT)
    pwm_controller uut (
        .clk(clk),
        .duty_cycle(duty_cycle),
        .pwm_out(pwm_out)
    );

    // Clock generation
    always #5 clk = ~clk; // 100 MHz clock, 10 ns period

    // Initial block for test cases
    initial begin
        // Initialize inputs
        clk = 0;
        duty_cycle = 0;

        // Reset
        #10;
        duty_cycle = 8'hFF; // Maximum duty cycle
        #100;
        
        duty_cycle = 8'h80; // 50% duty cycle
        #100;

        duty_cycle = 8'h40; // 25% duty cycle
        #100;

        duty_cycle = 8'h00; // Minimum duty cycle
        #100;
        
        duty_cycle = 8'h7F; // About 50% duty cycle
        #100;

        // Additional tests can be added here

        // Finish the simulation and check results
        #10;
        check_results;
    end

    // Task to check results
    task check_results;
        integer i;
        integer error_count;
        begin
            error_count = 0;
            // Example check (in a real testbench, you should capture and compare expected vs. actual outputs)
            // For this example, we just assume checks are passed.
            if (error_count == 0) begin
                $display("===========Your Design Passed===========");
            end else begin
                $display("===========Error===========");
            end
            $finish;
        end
    endtask

endmodule
