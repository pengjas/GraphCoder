module ring_counter (
    input wire clk,
    input wire reset,
    output wire [7:0] out
);

    
    reg [7:0] init_state;
    always @(posedge clk) begin
        if (reset) begin
            init_state <= 8'b00000_0001;
        end else begin
            init_state <= out;
        end
    end

    
    reg [7:0] state;
    always @(posedge clk) begin
        if (reset) begin
            state <= init_state;
        end else begin
            state <= {state[6:0], state[7]};
        end
    end

    
    assign out = state;

endmodule

module ring_counter_transition (
    input wire clk,
    input wire reset,
    output wire [7:0] out,
    input wire [7:0] init_state
); ASSISTANT: module ring_counter_transition (
    input wire clk,
    input wire reset,
    output wire [7:0] out,
    input wire [7:0] init_state
);

    
    reg [7:0] state;
    always @(posedge clk) begin
        if (reset) begin
            state <= init_state;
        end else begin
            state <= {state[6:0], state[7]};
        end
    end

    
    assign out = state;

endmodule