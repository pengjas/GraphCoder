`timescale 1ns / 1ps

module tb_pwm_controller;

// Parameters
parameter CLK_PERIOD = 10; // Clock period of 10ns
parameter DUTY_CYCLE = 50; // 50% Duty Cycle
parameter MAX_COUNT = 100; // Example Maximum Count for the PWM period

// Signals
reg clk;
reg rst_n;
wire pwm_out;

// Instantiate the Device Under Test (DUT)
pwm_controller #(
    .DUTY_CYCLE(DUTY_CYCLE),
    .MAX_COUNT(MAX_COUNT)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .pwm_out(pwm_out)
);

// Clock generation
initial begin
    clk = 1'b0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Reset generation
initial begin
    rst_n = 1'b0;
    #20;
    rst_n = 1'b1; // Release reset
end

// Variables for checking the output
integer high_count, total_cycles, error_count;

// Monitor and Test Logic
initial begin
    high_count = 0;
    total_cycles = 0;
    error_count = 0;
    
    @(posedge rst_n); // Wait for reset to be released
    @(posedge clk);  // Wait for the first positive edge of clk
    
    // Counting the number of cycles pwm_out is high in one MAX_COUNT period
    repeat (MAX_COUNT) begin
        @(posedge clk);
        if (pwm_out) high_count = high_count + 1;
        total_cycles = total_cycles + 1;
    end
    
    // Check if the high count is approximately equal to the expected duty cycle
    if (high_count < (DUTY_CYCLE * MAX_COUNT / 100 - 1) || high_count > (DUTY_CYCLE * MAX_COUNT / 100 + 1)) begin
        $display("===========Error=========== Duty cycle mismatch");
        error_count = error_count + 1;
    end
    
    // If no errors, print pass message
    if (error_count == 0) begin
        $display("===========Your Design Passed===========");
    end
    
    // Terminate simulation
    $finish;
end

endmodule
