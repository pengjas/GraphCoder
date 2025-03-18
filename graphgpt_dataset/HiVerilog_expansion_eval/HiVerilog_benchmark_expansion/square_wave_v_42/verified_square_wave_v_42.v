module pwm_controller(
    input clk,
    input [7:0] duty_cycle,
    output pwm_out
);

    wire [7:0] adjusted_duty;

    // Instance of Duty Cycle Controller
    duty_cycle_controller duty_ctrl (
        .clk(clk),
        .duty_cycle(duty_cycle),
        .adjusted_duty(adjusted_duty)
    );

    // Instance of Pulse Generator
    pulse_generator pulse_gen (
        .clk(clk),
        .adjusted_duty(adjusted_duty),
        .pwm_out(pwm_out)
    );

endmodule

module duty_cycle_controller(
    input clk,
    input [7:0] duty_cycle,
    output reg [7:0] adjusted_duty
);

    always @(posedge clk) begin
        adjusted_duty <= duty_cycle; // Placeholder for more complex logic if needed
    end

endmodule

module pulse_generator(
    input clk,
    input [7:0] adjusted_duty,
    output reg pwm_out
);
    reg [7:0] counter = 0;

    always @(posedge clk) begin
        if (counter < adjusted_duty)
            pwm_out <= 1;
        else
            pwm_out <= 0;

        counter <= (counter == 255) ? 0 : counter + 1;
    end

endmodule
