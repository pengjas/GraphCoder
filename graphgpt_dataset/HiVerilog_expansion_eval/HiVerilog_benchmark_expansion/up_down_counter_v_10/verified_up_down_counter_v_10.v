module binary_ripple_counter (
    input wire clk,
    input wire control_signal,
    output wire [31:0] count
);

    wire toggle;
    wire [31:0] next_count;

    // Instantiate the toggle logic
    toggle_logic u_toggle_logic (
        .control_signal(control_signal),
        .toggle(toggle)
    );

    // Instantiate the counter logic
    counter_logic u_counter_logic (
        .toggle(toggle),
        .current_count(count),
        .next_count(next_count)
    );

    // Instantiate the register to hold the count value
    counter_register u_counter_register (
        .clk(clk),
        .control_signal(control_signal),
        .next_count(next_count),
        .current_count(count)
    );

endmodule

module toggle_logic (
    input wire control_signal,
    output reg toggle
);

    always @(*) begin
        toggle = control_signal ? 1'b1 : 1'b0;
    end

endmodule

module counter_logic (
    input wire toggle,
    input wire [31:0] current_count,
    output reg [31:0] next_count
);

    always @(*) begin
        if (toggle) begin
            next_count = current_count + 1;
        end else begin
            next_count = 32'b0;
        end
    end

endmodule

module counter_register (
    input wire clk,
    input wire control_signal,
    input wire [31:0] next_count,
    output reg [31:0] current_count
);

    always @(posedge clk) begin
        if (control_signal == 1'b0) begin
            current_count <= 32'b0;
        end else begin
            current_count <= next_count;
        end
    end

endmodule
