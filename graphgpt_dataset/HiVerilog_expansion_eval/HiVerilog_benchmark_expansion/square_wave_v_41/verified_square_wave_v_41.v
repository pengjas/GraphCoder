module pwm_generator(
    input clk,
    input [7:0] duty_cycle,
    input [7:0] freq,
    output pwm_out
);

    wire [15:0] pwm_count;
    wire reset;

    // Instance of PWM counter module
    pwm_counter pc (
        .clk(clk),
        .reset(reset),
        .freq(freq),
        .count(pwm_count)
    );

    // Instance of PWM signal generator module
    pwm_signal ps (
        .clk(clk),
        .count(pwm_count),
        .duty_cycle(duty_cycle),
        .freq(freq),
        .pwm_out(pwm_out)
    );

    // Reset logic
    assign reset = (freq == 0);

endmodule

module pwm_counter(
    input clk,
    input reset,
    input [7:0] freq,
    output reg [15:0] count
);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 0;
        end else begin
            if (count >= freq - 1) begin
                count <= 0;
            end else begin
                count <= count + 1;
            end
        end
    end

endmodule

module pwm_signal(
    input clk,
    input [15:0] count,
    input [7:0] duty_cycle,
    input [7:0] freq,
    output reg pwm_out
);

    always @(posedge clk) begin
        if (count < (freq * duty_cycle / 100)) begin
            pwm_out <= 1;
        end else begin
            pwm_out <= 0;
        end
    end

endmodule