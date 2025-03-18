module pwm_controller(
    input clk,
    input rst_n,
    output pwm_out
);
    // Parameters for PWM configuration
    parameter DUTY_CYCLE = 50; // 50% duty cycle
    parameter MAX_COUNT = 100; // Full cycle count

    wire [6:0] count_value; // assuming 7-bit for 100 max count
    wire pwm_signal;

    // Instantiate the counter module
    counter #(.MAX_COUNT(MAX_COUNT)) u_counter (
        .clk(clk),
        .rst_n(rst_n),
        .count(count_value)
    );

    // Instantiate the comparator module
    comparator #(.DUTY_CYCLE(DUTY_CYCLE), .MAX_COUNT(MAX_COUNT)) u_comparator (
        .count(count_value),
        .pwm(pwm_signal)
    );

    // Assign the comparator output to the PWM output
    assign pwm_out = pwm_signal;

endmodule

module counter(
    input clk,
    input rst_n,
    output reg [6:0] count
);
    parameter MAX_COUNT = 100;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            if (count >= MAX_COUNT - 1) begin
                count <= 0;
            end else begin
                count <= count + 1;
            end
        end
    end
endmodule

module comparator(
    input [6:0] count,
    output reg pwm
);
    parameter DUTY_CYCLE = 50;
    parameter MAX_COUNT = 100;

    always @(*) begin
        if (count < (MAX_COUNT * DUTY_CYCLE / 100)) begin
            pwm = 1'b1;
        end else begin
            pwm = 1'b0;
        end
    end
endmodule
